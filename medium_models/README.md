# HiZOO on Medium-sized Masked Language Models

This part of the code is for HiZOO experiments on RoBERTa-large.

## Installation

Please install the latest versions of PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)) and 
Transformers (`transformers`). This code is tested on `torch==2.1.0.dev20230514+cu118` 
and `transformers==4.28.1` with Python 3.9.7.

## Prepare the data

The datasets are from [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar) 
by [MeZO](https://github.com/princeton-nlp/MeZO/tree/main). 
Please download it and extract the files to `./data/original`.

Then use the following command (in the `medium_models` folder) to generate the data we need:

```bash
for K in 16 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, 
we use the default options: we take `K=16` and `K=512` and take 5 different seeds 
of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot-1k-test`. 
In the directory of each dataset, there will be folders named as `$K-$SEED` indicating 
different dataset samples.

## Usage

Use `run.py` for all functions and refer to `run.py` for the usage of all arguments.
```bash
python run.py {ARGUMENTS}
```

To reproduce our results in the paper, you can run them directly with the following commands:

```bash

# HiZOO
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-6 EPS=1e-3 HESSIAN_SMOOTH_TYPE=constant1e-6 MODEL=roberta-large bash HiZOO.sh

# HiZOO + prefix-tuning
TASK=SST-2 K=16 SEED=42 BS=64 LR=5e-2 EPS=1e-1 HESSIAN_SMOOTH_TYPE=constant1e-6 MODEL=roberta-large EXTRA_TAG=prefix bash HiZOO.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# HiZOO + LoRA
TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 WD=0.1 HESSIAN_SMOOTH_TYPE=constant1e-10 MODEL=roberta-large EXTRA_TAG=lora bash HiZOO.sh --apply_lora --lora_r 8 --lora_alpha 16
```

Our recommended hyperparameter search range for RoBERTa-large can be found in appendix of our paper.