# Second-Order Fine-Tuning without Pain for LLMs: a Hessian Informed Zeroth-Order Optimizer(ICLR 2025)


In this work, we propose a diagonal
hessian-informed zeroth-order optimizer(HiZOO)
without computing first-order or second-order
derivatives. To our knowledge, this is the first
work that leverages hessian to enhance zeroth-order optimizer for fine-tuning LLMs. What’s
more, HiZOO avoids the heavy memory cost
brought by backpropagation while only increases
one forward pass per step. Extensive experiments
on various models(350M∼66B parameters) indicate that HiZOO efficiently improves model convergence, reducing training steps and enhancing
model accuracy.


## Installation

```bash
conda create -n HiZOO python==3.9.19
conda activate HiZOO
pip install -r requirements.txt
```

This environment can support the **OPT**, **LLaMA**, **Phi** and other latest models.

## Usage

Use `run.py` for all functions (zero-shot/ICL/fine-tuning/MeZO/HiZOO):

```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments.

We provide example script below for reproducing our experiments. All our examples sample 1,000 
training examples, 500 validation examples, and 1,000 testing examples. 

```bash
# HiZOO (full-parameter fine-tune OPT-13B on CB dataset)
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=WSC MODE=ft LR=1e-6 EPS=1e-3 HESSIAN_SMOOTH_TYPE=constant1e-8 bash HiZOO.sh

```

