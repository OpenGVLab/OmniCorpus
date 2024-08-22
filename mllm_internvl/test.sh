CKPT_ROOT=".."
RESULT_ROOT="./results"
CKPT_FNAMES=(
    "checkpoint-10200"
)
mkdir -p $RESULT_ROOT

for CKPT_FNAME in "${CKPT_FNAMES[@]}"; do
    set -x
    PYTHONPATH="/mnt/petrelfs/liqingyun/OmniCorpus_internal/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 0 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_0shots-multi-rounds.result"
    PYTHONPATH="/mnt/petrelfs/liqingyun/OmniCorpus_internal/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 1 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_1shots-multi-rounds.result"
    PYTHONPATH="/mnt/petrelfs/liqingyun/OmniCorpus_internal/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 2 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_2shots-multi-rounds.result"
    PYTHONPATH="/mnt/petrelfs/liqingyun/OmniCorpus_internal/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 4 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_4shots-multi-rounds.result"
    PYTHONPATH="/mnt/petrelfs/liqingyun/OmniCorpus_internal/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 8 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_8shots-multi-rounds.result"
    PYTHONPATH="/mnt/petrelfs/liqingyun/OmniCorpus_internal/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 0 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 --zero-shot-add-text-shots 2 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_trick0shot-multi-rounds.result"
done
