# Second-Order Fine-Tuning without Pain for LLMs: a Hessian Informed Zeroth-Order Optimizer(HiZOO)

This is the implementation for the paper **Second-Order Fine-Tuning without Pain for LLMs: a Hessian Informed Zeroth-Order Optimizer**. 

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

## Reproduce our paper results

For reproducing RoBERTa-large experiments, please refer to the
[medium_models](https://github.com/Yanjun-Zhao/HiZOO/tree/main/medium_models) folder.

For autoregressive LM (OPT) experiments, please refer to the
[large_models](https://github.com/Yanjun-Zhao/HiZOO/tree/main/large_models) folder.

## Acknowledgment

This project is built upon the foundation laid by [MeZO: Fine-Tuning Language Models with 
Just Forward Passes](https://github.com/princeton-nlp/MeZO). 
The original code from their project is licensed under the [MIT License](https://github.com/princeton-nlp/MeZO/blob/main/LICENSE).