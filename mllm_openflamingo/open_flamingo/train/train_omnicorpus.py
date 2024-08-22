""" Main training script """

import argparse
import glob
import os
import random

import numpy as np
import torch
import wandb
from data import get_data
from distributed import world_info_from_env
from train_utils import (
    train_one_epoch_with_accelerator,
    get_checkpoint,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

import deepspeed
try:
    from transformers.integrations import deepspeed_config
except:
    from transformers.deepspeed import deepspeed_config

from open_flamingo import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument(
        "--lm_path", 
        default="/mnt/petrelfs/share_data/wangwenhai/llm/Meta-Llama-3-8B", 
        type=str,
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/mnt/petrelfs/share_data/wangwenhai/llm/Meta-Llama-3-8B", 
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=4,
        help="how often to add a cross-attention layer after each transformer layer",
    )

    # training args
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo_llama8b_obelics_laionen2b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Whether to resume from checkpoint, if set True, will load models from --external_save_dir",
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--batch_size_interleaved", type=int, default=128)
    parser.add_argument("--batch_size_laion", type=int, default=128)
    parser.add_argument("--max_tokens_interleaved", type=int, default=256)
    parser.add_argument("--max_tokens_laion", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_interleaved", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_laion", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples (train_num_samples_interleaved, train_num_samples_laion), not a pass through the entire dataset",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--freeze_lm_embeddings",
        action="store_true",
        help="if True, we freeze the LM embeddings during training. Otherwise, we train the <image> and <|endofchunk|> embeddings.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help="checkpointing every n steps",
    )
    parser.add_argument(
        "--special_checkpointing_steps",
        type=int,
        nargs='+',
        default=[],
        help="checkpointing at several special steps",
    )

    # data args
    parser.add_argument(
        "--obelics_data_path", 
        type=str,
        default="s3://public-dataset/OBELISC/jsonl/", 
    )
    parser.add_argument(
        "--obelics_image_path", 
        type=str,
        default="s3://public-dataset/OBELISC/raw-images/", 
    )
    parser.add_argument(
        "--laionen2b_data_path", 
        type=str,
        default="/mnt/petrelfs/share_data/wangwenhai/to_gaozhangwei/datasets/laion5b/LaionEn", 
    )
    parser.add_argument(
        "--laionen2b_image_path", 
        type=str,
        default="langchao:s3://LAION-5B-P/LAION-5B/", 
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_interleaved", type=int, default=140529700)
    parser.add_argument("--train_num_samples_laion", type=int, default=140529700)
    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument(
        "--mmc4_max_num_images",
        default=6,
        type=int,
        help="max number of images per sequence in mmc4 / chatgpt",
    )

    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(  # TODO
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--resume_from_steps",
        type=int,
        default=0,
    )

    parser.add_argument(  # TODO
        "--load_from_pretrained",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    
    parser.add_argument(
        "--use_iterable_dataset",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_clip_sim",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    # Validate args
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
    # get distributed training info
    args.local_rank, args.rank, args.world_size = world_info_from_env()    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                              mixed_precision="bf16")
    print((args.rank, args.local_rank, args.world_size))
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size_laion + args.batch_size_interleaved
    device_id = accelerator.device   

    assert (args.train_num_samples_laion // args.batch_size_laion) == (
        args.train_num_samples_interleaved // args.batch_size_interleaved
    ), "number of samples per epoch must be equal for mmc4 and laion"
    
    random_seed(args.seed)

    # Initialize model
    # with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )
    
    if args.load_from_pretrained is not None:
        checkpoint = torch.load(args.load_from_pretrained, map_location="cpu")
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        print(model.load_state_dict(checkpoint, strict=False))
    
    model.train()
    
    random_seed(args.seed, args.rank)

    # control the number of samples
    total_steps = (args.train_num_samples_interleaved) // (args.batch_size_interleaved * args.world_size)
    args.train_num_samples_interleaved = total_steps * args.batch_size_interleaved * args.world_size
    args.train_num_samples_laion = total_steps * args.batch_size_laion * args.world_size
    
    # Initialize data loaders
    laion_dataset = get_data(args, image_processor, tokenizer, "laionen2b")
    if args.use_clip_sim:
        obelics_dataset = get_data(args, image_processor, tokenizer, "omnicorpus_w_similarities")
    else:
        obelics_dataset = get_data(args, image_processor, tokenizer, "omnicorpus")
    total_training_steps = (
        (args.train_num_samples_interleaved) // (args.batch_size_interleaved * args.world_size)
    ) * args.num_epochs  # TODO FIXME

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    
    # Initialize optimizer
    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]
    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)
    
    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,  # TODO FIXME
            num_training_steps=total_training_steps,  # TODO FIXME
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,  # TODO FIXME
            num_training_steps=total_training_steps,  # TODO FIXME
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
        
    # Initialize logging
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        
    # Load model checkpoint  # TODO FIXME This checkpoint mechanism is not good
    resume_from_epoch = 0
    resume_from_checkpoint_path = None
    # check if a checkpoint exists for this run
    args.external_save_dir = os.path.join(args.external_save_dir, args.run_name) if args.external_save_dir else args.run_name
    if os.path.exists(f"{args.external_save_dir}") and args.resume_from_checkpoint is True:
        assert args.resume_from_steps == 0, "Cannot resume from both steps and epoch"
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            resume_from_checkpoint_path = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}.")

    if os.path.exists(f"{args.external_save_dir}") and args.resume_from_steps > 0:
        resume_from_checkpoint_path = f"{args.external_save_dir}/checkpoint_steps{args.resume_from_steps}.pt"
        
    if resume_from_checkpoint_path is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint_path}")
        checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], False)
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if args.resume_from_checkpoint:
            resume_from_epoch = checkpoint["epoch"] + 1
        else:
            resume_from_epoch = checkpoint["epoch"]

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    
    # # Initialize gradient checkpointing  # TODO
    # if args.gradient_checkpointing:
    #     non_reentrant_wrapper = functools.partial(
    #         checkpoint_wrapper,
    #         offload_to_cpu=True,
    #         checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    #     )
    #     apply_activation_checkpointing(
    #         model,
    #         checkpoint_wrapper_fn=non_reentrant_wrapper,
    #         check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
    #         and not isinstance(m, FSDP)
    #         and not isinstance(m, CheckpointWrapper),
    #     )

    # Start training!
    model.train()
    print(f"Start running training on rank {args.rank}.")
    for epoch in range(resume_from_epoch, args.num_epochs):
        laion_dataset.set_epoch(epoch)
        laion_loader = laion_dataset.dataloader
        obelics_dataset.set_epoch(epoch)
        obelics_loader = obelics_dataset.dataloader

        train_one_epoch_with_accelerator(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            laion_loader=laion_loader,
            mmc4_loader=obelics_loader,
            device_id=device_id,
            accelerator=accelerator,
            wandb=wandb,
            resume_from_steps=args.resume_from_steps,
        )
        if args.rank == 0:
            os.makedirs(args.external_save_dir, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(unwrapped_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_epoch{epoch}.pt")
            accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_epoch{epoch}.pt")
            # # save the config
            # unwrapped_model.config.save_pretrained(args.external_save_dir)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(f"{args.external_save_dir}/checkpoint_epoch{epoch-1}.pt")

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if args.rank == 0:
        os.makedirs(args.external_save_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(
            get_checkpoint(model=unwrapped_model),
            f"{args.external_save_dir}/final_weights.pt",
        )
        # save the config
        unwrapped_model.config.save_pretrained(args.external_save_dir)

        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.external_save_dir}/final_weights.pt")
        if args.save_hf_model:
            unwrapped_model.save_pretrained(f"{args.external_save_dir}")


if __name__ == "__main__":
    main()
