import time
from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
import os
import wandb
from accelerate import Accelerator
from einops import rearrange


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    laion_loader,
    mmc4_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # setup loaders
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_interleaved = mmc4_loader.num_batches
    assert (
        num_batches_per_epoch_laion == num_batches_per_epoch_interleaved
    ), f"Number of batches in laion {num_batches_per_epoch_laion} and mmc4 {num_batches_per_epoch_interleaved} datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_interleaved
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)  # NOTE
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)  # NOTE

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    update_total_losses_list = True

    # loop through dataloader
    for num_steps, (batch_laion, batch_interleaved) in tqdm(
        enumerate(zip(laion_loader, mmc4_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        if update_total_losses_list:
            total_losses = []
            update_total_losses_list = False

        #### LAION FORWARD PASS ####
        images = batch_laion[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        
        if isinstance(batch_laion[1][0], torch.Tensor) and isinstance(batch_laion[1][1], torch.Tensor):
            input_ids = batch_laion[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = batch_laion[1][1].to(
                device_id, dtype=cast_dtype, non_blocking=True
            )
        else:
            input_ids = [batch_laion[1][i][0] for i in range(len(batch_laion[1]))]
            input_ids = torch.stack(input_ids)
            input_ids = input_ids.to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = [batch_laion[1][i][1] for i in range(len(batch_laion[1]))]
            attention_mask = torch.stack(attention_mask)
            attention_mask = attention_mask.to(device_id, dtype=cast_dtype, non_blocking=True)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        divided_loss_laion = loss_laion / args.gradient_accumulation_steps  # NOTE
        (divided_loss_laion * args.loss_multiplier_laion).backward()
        total_losses.append(args.loss_multiplier_laion * divided_loss_laion)

        #### MMC4 FORWARD PASS ####
        images = batch_interleaved[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        input_ids = torch.stack([x[0] for x in batch_interleaved[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_interleaved[1]]).squeeze(1)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss_interleaved = model(
                vision_x=images,
                lang_x=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                labels=labels,
            )[0]

            # if loss is nan, skip this batch
            # this hack of skipping the batch is not FSDP-compatible
            if torch.isnan(loss_interleaved):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        divided_loss_interleaved = loss_interleaved / args.gradient_accumulation_steps  # NOTE
        (divided_loss_interleaved * args.loss_multiplier_interleaved).backward()
        total_losses.append(args.loss_multiplier_interleaved * divided_loss_interleaved)

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(
                zero_mask[endofchunk_token_id]
            )
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            #### Collect MMC4/LAION Loss Info ####
            total_loss_sum = sum(total_losses)
            mean_loss = total_loss_sum / len(total_losses)
            update_total_losses_list = True
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )
                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    / step_time_m.val
                )
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()
                
                wandb.log(
                    {
                        "loss_interleaved": loss_interleaved.item(),
                        "loss_laion": loss_laion.item(),
                        "mean_loss": mean_loss.item(),
                        "global_step": global_step,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss LAION: {loss_laion.item():.3f} // Loss MMC4: {loss_interleaved.item():.3f} // Mean Loss: {mean_loss.item():.3f}"
            )
            
            
def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict
            
            
def train_one_epoch_with_accelerator(
    args,
    model,
    epoch,
    mmc4_loader,
    laion_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    accelerator: Accelerator,
    wandb,
    resume_from_steps=0,
):
    """
    Mostly copied from Otter
    """
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_interleaved = mmc4_loader.num_batches
    assert (
        num_batches_per_epoch_laion == num_batches_per_epoch_interleaved
    ), f"Number of batches in laion {num_batches_per_epoch_laion} and mmc4 {num_batches_per_epoch_interleaved} datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_interleaved
    total_training_steps = num_batches_per_epoch * args.num_epochs

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()
    
    # laion_iter = iter(laion_loader)
    # mmc4_iter = iter(mmc4_loader)
    if resume_from_steps > 0:
        mmc4_loader.sampler.jump(resume_from_steps)
        laion_loader.sampler.jump(resume_from_steps)
        for _ in range(resume_from_steps):
            lr_scheduler.step()

    # loop through dataloader
    for num_steps, (batch_laion, batch_interleaved) in tqdm(
        enumerate(zip(laion_loader, mmc4_loader)),
    # for num_steps in tqdm(
    #     range(resume_from_steps + 1, num_batches_per_epoch),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch) + resume_from_steps,
    ):
        # batch_laion = next(laion_iter)
        # batch_interleaved = next(mmc4_iter)
        
        data_time_m.update(time.time() - end)
        num_steps += resume_from_steps
        global_step = num_steps + epoch * num_batches_per_epoch
        total_losses = []

        #### LAION FORWARD PASS ####
        images = batch_laion[0].to(dtype=torch.bfloat16, device=device_id, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        
        input_ids = [batch_laion[1][i][0] for i in range(len(batch_laion[1]))]
        input_ids = torch.stack(input_ids)
        input_ids = input_ids.to(device=device_id, non_blocking=True)
        
        attention_mask = [batch_laion[1][i][1] for i in range(len(batch_laion[1]))]
        attention_mask = torch.stack(attention_mask)
        attention_mask = attention_mask.to(device_id, non_blocking=True)

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        with accelerator.accumulate(model), accelerator.autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        #### LAION BACKWARD ####
        accelerator.backward(args.loss_multiplier_laion * loss_laion)
        total_losses.append(args.loss_multiplier_laion * loss_laion)

        #### MMC4 FORWARD PASS ####
        images = batch_interleaved[0].to(dtype=torch.bfloat16, device=device_id, non_blocking=True)
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        input_ids = torch.stack([x[0] for x in batch_interleaved[1]]).squeeze(1).to(device=device_id, non_blocking=True)
        attention_mask = torch.stack([x[1] for x in batch_interleaved[1]]).squeeze(1).to(device=device_id, non_blocking=True)

        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id:
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != media_token_id:
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        with accelerator.accumulate(model), accelerator.autocast():
            loss_interleaved = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
            
            # if loss is nan, skip this batch
            # this hack of skipping the batch is not FSDP-compatible
            if torch.isnan(loss_interleaved):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        #### MMC4 BACKWARD ####
        accelerator.backward(args.loss_multiplier_interleaved * loss_interleaved)
        total_losses.append(args.loss_multiplier_interleaved * loss_interleaved)
        #### Collect MMC4/LAION Loss Info ####
        total_loss_sum = sum(total_losses)
        mean_loss = total_loss_sum / len(total_losses)
        # accelerator.backward(total_loss_sum.to(device_id))

        # def mask_embedding(m):
        #     if m.weight.requires_grad:
        #         print(m)
        #         zero_mask = torch.zeros_like(m.weight.grad)
        #         # zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
        #         zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
        #         zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
        #         m.weight.grad = m.weight.grad * zero_mask
        # if not args.freeze_lm_embeddings:
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     # For LlamaForCausalLM
        #     unwrapped_model.lang_encoder.model.embed_tokens.apply(mask_embedding)
        #     unwrapped_model.lang_encoder.lm_head.apply(mask_embedding)
        
        def apply_grad_mask(grad):
            zero_mask = torch.zeros_like(grad)
            # 假设 media_token_id 和 endofchunk_token_id 已经被定义
            zero_mask[media_token_id] = 1.0
            zero_mask[endofchunk_token_id] = 1.0
            return grad * zero_mask
        if not args.freeze_lm_embeddings:
            unwrapped_model = accelerator.unwrap_model(model)
            param = unwrapped_model.lang_encoder.get_input_embeddings().weight
            hook_handle = param.register_hook(apply_grad_mask)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if not args.freeze_lm_embeddings:
            hook_handle.remove()

        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()

        if accelerator.sync_gradients:
            if args.rank == 0 and args.report_to_wandb:
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )
                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    / step_time_m.val
                )
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_interleaved": loss_interleaved.item(),
                        "loss_laion": loss_laion.item(),
                        "mean_loss": mean_loss.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss LAION: {loss_laion.item():.3f} // Loss MMC4: {loss_interleaved.item():.3f}"
            )
        
        # Add a process on saving checkpoints during pretraining
        if any([
            (num_steps + 1) % args.checkpointing_steps == 0, 
            (num_steps + 1) in getattr(args, "special_checkpointing_steps", []),
        ]) and args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(unwrapped_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt")
            accelerator.save(
                checkpoint_dict,
                f"{args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt",
            )
            # save the config
            print(f"Saving config to {args.external_save_dir}/config.json")
            # unwrapped_model.config.save_pretrained(args.external_save_dir)
            if args.delete_previous_checkpoint:
                if (num_steps + 1) // args.checkpointing_steps >= 2:
                    previous_checkpoint_path = f"{args.external_save_dir}/checkpoint_steps{num_steps + 1 - args.checkpointing_steps}.pt"
                    if os.path.exists(previous_checkpoint_path):
                        os.remove(previous_checkpoint_path)
                        
                        
def train_one_epoch_no_laion_with_accelerator(
    args,
    model,
    epoch,
    mmc4_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    accelerator: Accelerator,
    wandb,
):
    """
    Mostly copied from Otter
    """
    num_batches_per_epoch_interleaved = mmc4_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_interleaved
    total_training_steps = num_batches_per_epoch * args.num_epochs

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, batch_interleaved in tqdm(enumerate(mmc4_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        #### MMC4 FORWARD PASS ####
        images = batch_interleaved[0].to(dtype=torch.bfloat16, device=device_id, non_blocking=True)
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        input_ids = torch.stack([x[0] for x in batch_interleaved[1]]).squeeze(1).to(device=device_id, non_blocking=True)
        attention_mask = torch.stack([x[1] for x in batch_interleaved[1]]).squeeze(1).to(device=device_id, non_blocking=True)

        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id:
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != media_token_id:
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)
        
        # import ipdb; ipdb.set_trace()
        # labels_debug = labels.clone()
        # labels_debug[labels_debug == -100] = tokenizer.pad_token_id
        # for i in range(len(input_ids)): print(tokenizer.decode(input_ids[i])); print(tokenizer.decode(labels_debug[i]))
        # for i in range(len(input_ids)):
        #     os.makedirs("tmp", exist_ok=True)
        #     with open(f"tmp/tmp_input_ids_1.txt", "w") as f:
        #         f.write(tokenizer.decode(input_ids[i]))
        #     with open(f"tmp/tmp_labels_debug_1.txt", "w") as f:
        #         f.write(tokenizer.decode(labels_debug[i]))

        with accelerator.accumulate(model), accelerator.autocast():
            loss_interleaved = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
            
            # if loss is nan, skip this batch
            # this hack of skipping the batch is not FSDP-compatible
            if torch.isnan(loss_interleaved):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        #### MMC4 BACKWARD ####
        accelerator.backward(args.loss_multiplier_interleaved * loss_interleaved)

        # def mask_embedding(m):
        #     if m.weight.requires_grad:
        #         print(m)
        #         zero_mask = torch.zeros_like(m.weight.grad)
        #         # zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
        #         zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
        #         zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
        #         m.weight.grad = m.weight.grad * zero_mask

        # if not args.freeze_lm_embeddings:
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     # For LlamaForCausalLM
        #     unwrapped_model.lang_encoder.model.embed_tokens.apply(mask_embedding)
        #     unwrapped_model.lang_encoder.lm_head.apply(mask_embedding)
        
        def apply_grad_mask(grad):
            zero_mask = torch.zeros_like(grad)
            # 假设 media_token_id 和 endofchunk_token_id 已经被定义
            zero_mask[media_token_id] = 1.0
            zero_mask[endofchunk_token_id] = 1.0
            return grad * zero_mask
        if not args.freeze_lm_embeddings:
            unwrapped_model = accelerator.unwrap_model(model)
            param = unwrapped_model.lang_encoder.get_input_embeddings().weight
            hook_handle = param.register_hook(apply_grad_mask)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if not args.freeze_lm_embeddings:
            hook_handle.remove()

        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()

        if accelerator.sync_gradients:
            if args.rank == 0 and args.report_to_wandb:
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )
                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    / step_time_m.val
                )
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {"loss_interleaved": loss_interleaved.item(), "global_step": global_step // args.gradient_accumulation_steps},
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss MMC4: {loss_interleaved.item():.3f}"
            )
        
        # Add a process on saving checkpoints during pretraining
        if any([
            (num_steps + 1) % args.checkpointing_steps == 0, 
            (num_steps + 1) in getattr(args, "special_checkpointing_steps", []),
        ]) and args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(unwrapped_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt")
            accelerator.save(
                checkpoint_dict,
                f"{args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt",
            )
            # save the config
            print(f"Saving config to {args.external_save_dir}/config.json")
            # unwrapped_model.config.save_pretrained(args.external_save_dir)
            if args.delete_previous_checkpoint:
                if (num_steps + 1) // args.checkpointing_steps >= 2:
                    previous_checkpoint_path = f"{args.external_save_dir}/checkpoint_steps{num_steps + 1 - args.checkpointing_steps}.pt"
                    if os.path.exists(previous_checkpoint_path):
                        os.remove(previous_checkpoint_path)
                        
                        
def train_one_epoch_only_laion_with_accelerator(
    args,
    model,
    epoch,
    laion_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    accelerator: Accelerator,
    wandb,
):
    """
    Mostly copied from Otter
    """
    num_batches_per_epoch = laion_loader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, batch_laion in tqdm(enumerate(laion_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        total_losses = []

        #### LAION FORWARD PASS ####
        images = batch_laion[0].to(dtype=torch.bfloat16, device=device_id, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        
        input_ids = [batch_laion[1][i][0] for i in range(len(batch_laion[1]))]
        input_ids = torch.stack(input_ids)
        input_ids = input_ids.to(device=device_id, non_blocking=True)
        
        attention_mask = [batch_laion[1][i][1] for i in range(len(batch_laion[1]))]
        attention_mask = torch.stack(attention_mask)
        attention_mask = attention_mask.to(device_id, non_blocking=True)

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        with accelerator.accumulate(model), accelerator.autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        #### LAION BACKWARD ####
        accelerator.backward(args.loss_multiplier_laion * loss_laion)
        total_losses.append(args.loss_multiplier_laion * loss_laion)

        # def mask_embedding(m):
        #     if m.weight.requires_grad:
        #         print(m)
        #         zero_mask = torch.zeros_like(m.weight.grad)
        #         # zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
        #         zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
        #         zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
        #         m.weight.grad = m.weight.grad * zero_mask
        # if not args.freeze_lm_embeddings:
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     # For LlamaForCausalLM
        #     unwrapped_model.lang_encoder.model.embed_tokens.apply(mask_embedding)
        #     unwrapped_model.lang_encoder.lm_head.apply(mask_embedding)

        def apply_grad_mask(grad):
            zero_mask = torch.zeros_like(grad)
            # 假设 media_token_id 和 endofchunk_token_id 已经被定义
            zero_mask[media_token_id] = 1.0
            zero_mask[endofchunk_token_id] = 1.0
            return grad * zero_mask
        if not args.freeze_lm_embeddings:
            unwrapped_model = accelerator.unwrap_model(model)
            param = unwrapped_model.lang_encoder.get_input_embeddings().weight
            hook_handle = param.register_hook(apply_grad_mask)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if not args.freeze_lm_embeddings:
            hook_handle.remove()

        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()

        if accelerator.sync_gradients:
            if args.rank == 0 and args.report_to_wandb:
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )
                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_interleaved
                    / step_time_m.val
                )
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_laion": loss_laion.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss LAION: {loss_laion.item():.3f}"
            )
        
        # Add a process on saving checkpoints during pretraining
        if any([
            (num_steps + 1) % args.checkpointing_steps == 0, 
            (num_steps + 1) in getattr(args, "special_checkpointing_steps", []),
        ]) and args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

            unwrapped_model = accelerator.unwrap_model(model)
            print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt")
            accelerator.save(
                {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(unwrapped_model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                },
                f"{args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt",
            )
            # save the config
            print(f"Saving config to {args.external_save_dir}/config.json")
            # unwrapped_model.config.save_pretrained(args.external_save_dir)
            if args.delete_previous_checkpoint:
                if (num_steps + 1) // args.checkpointing_steps >= 2:
                    previous_checkpoint_path = f"{args.external_save_dir}/checkpoint_steps{num_steps + 1 - args.checkpointing_steps}.pt"
                    if os.path.exists(previous_checkpoint_path):
                        os.remove(previous_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)

    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")
