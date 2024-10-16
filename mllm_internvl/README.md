# InternVL-Chat Pre-trained with OmniCorpus

This folder contains the interleaved image-text implementation for InternVL-Chat. Welcome to use [InternVL-family](https://github.com/OpenGVLab/InternVL). Our checkpoint is availiable [here](https://huggingface.co/Qingyun/OmniCorpus-InternVL).

## 🛠️ Installation

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n internvl_interleaved python=3.9.0 -y
  conda activate internvl_interleaved
  ```
- Install `PyTorch>=2.0` and `torchvision>=0.15.2` with `CUDA>=11.6`:

  For examples, to install `torch==2.0.1` with `CUDA==11.8`:

  ```bash
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  # or
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```
- Install requirements:

  ```
  pip install -r requirements.txt
  pip install open_flamingo[eval]
  pip install open_clip_torch open_flamingo --no-deps
  ```
- Clone and install InternVL:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  cd InternVL/internvl_chat
  pip install -e .
  ```
- Install `flash-attn==2.3.6`:

  ```bash
  pip install flash-attn==2.3.6 --no-build-isolation
  ```

  Alternatively you can compile from source:

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v2.3.6
  python setup.py install
  ```
- Install `transformers==4.37.2`:

  ```bash
  pip install transformers==4.37.2
  ```
- Download for nltk with:

  ```
  python -c "import nltk;nltk.download('punkt');nltk.download('punkt_tab');nltk.download('wordnet');nltk.download('averaged_perceptron_tagger');nltk.download('averaged_perceptron_tagger_eng')"
  ```

> Note you may need to delete spice in pycocoevalcap if is useless `$CONDA_HOME/envs/xxx/lib/pythonX.X/site-packages/pycocoevalcap/eval.py`

## Usage of Pre-trained MLLMs Evaluation Tools

The code is modified from [OpenFlamingo Evaluation](https://github.com/mlfoundations/open_flamingo/tree/main/open_flamingo/eval)

script `eval_pretrain/evaluate.py` is the python running script of evaluating pretraining. script `eval_pretrain/evaluate_with_slurm.sh` is the shell running script of distributed evaluation for slurm user.

First, you need to modify the path in `ds_collections` of `eval_pretrain/evaluate.py`.

- `--model` : choice ["open_flamingo", "internvl_chat"], model type
- `--result_file`: file path to save a dict of results
- `--batch_size`: we now only support batch_size=1 for internvl_chat
- `--shots`: provide multiple shots like `--shot 0 4 8 16 32`
- `--dataset`: provide multiple datasets like `--datasets flickr ok_vqa textvqa coco vqav2 vizwiz`
- `--num_trials` and `--trial_seeds` set the randomness of selecting in-context samples, the mean value will be calculated.
- `--rices` whether to use RICES for evaluation. If False, uses random demonstrations.
- `--zero-shot-add-text-shots` whether to use pure text examples when evaluating zero-shot performance, usually `--zero-shot-add-text-shots 2`
- `--chat-few-shot-style`: choice ["multi", "single"]. whether to put all examples in the first question or conduct as a multi-rounds conversation.
- For `internvl_chat`: `--checkpoint`, `--load_in_8bit`, `--dynamic`, `--max_num` must be provided.

**Example**:

```
CKPT_ROOT="."
RESULT_ROOT="./results"
CKPT_FNAMES=(
    "checkpoint"
)
mkdir -p $RESULT_ROOT

for CKPT_FNAME in "${CKPT_FNAMES[@]}"; do
    set -x
    PYTHONPATH="path_to_OmniCorpus/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 0 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_0shots-multi-rounds.result"
    PYTHONPATH="path_to_OmniCorpus/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 1 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_1shots-multi-rounds.result"
    PYTHONPATH="path_to_OmniCorpus/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 2 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_2shots-multi-rounds.result"
    PYTHONPATH="path_to_OmniCorpus/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 4 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_4shots-multi-rounds.result"
    PYTHONPATH="path_to_OmniCorpus/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 8 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_8shots-multi-rounds.result"
    PYTHONPATH="path_to_OmniCorpus/mllm_internvl/InternVL/internvl_chat"$PYTHONPATH \
        srun bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 0 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 --zero-shot-add-text-shots 2 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_trick0shot-multi-rounds.result"
done
```
