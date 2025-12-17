# K2-V2: A 360-Open, Reasoning-Enhanced Open Foundation Model

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/cs.LG-2512.06201-b31b1b?logo=arxiv&logoColor=#B31B1B)](https://www.arxiv.org/abs/2512.06201)

[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base%20Model-K2--V2-blue)](https://huggingface.co/LLM360/K2-V2)
[![SFT Model](https://img.shields.io/badge/%F0%9F%A4%97%20SFT%20Model-K2--V2-blue)](https://huggingface.co/LLM360/K2-V2-Instruct)

[![Pre-Training Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data%20PreTrain-K2--V2-blue)](https://huggingface.co/datasets/LLM360/TxT360)
[![Mid-Training Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data%20MidTrain-K2--V2-blue)](https://huggingface.co/datasets/LLM360/TxT360-Midas)


This repository contains the codebase for **K2-V2**, a fully open-source, reasoning-enhanced foundation model. K2-V2 is designed with a "360-Open" philosophy, providing full transparency across the training pipeline—including data processing, pre-training, mid-training (annealing), and supervised fine-tuning (SFT).


# Setup

The core training scripts follows [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). We recommend using a Docker container for reproducibility, but local environments can be set up, by instoalling the latest pytorch, cuda, nccl and NVdia Apex.

We follow the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) setup, and use the [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) with DGX nodes for environment setup. 

# Training Scripts
- Pre-training

The master pre-training script is located under `pre-train/k2v2_70b_400nodes_120bsz.sh`. Several data paths are marked to be adjusted based on your cluster environment.

- Mid-training

The mid-training script is located in the [mid-train](mid-train). It is similar to pre-training, with a key difference on context length across the 4 stages. The data can be obtained through [TxT360-Midas](https://huggingface.co/datasets/LLM360/TxT360-Midas), the datasets are organized as subsets, corresponding to the stages.

- Training Monitor

We develop a light-weight [training monitor](monitor/README.md) for our large scale training jobs. Note that the monitor isn't necessary for your training jobs. It is shared in case it is useful.

- Supervised Fine-Tuning (SFT)

Instructions are located in the [sft](sft). This script conducts a simple SFT using the [TxT360-3efforts](https://huggingface.co/datasets/LLM360/TxT360-3efforts) dataset. Since the dataset is organized as typical chat templates, feel free to use other SFT library you find handy.

- [Evaluation](https://github.com/LLM360/eval360)

Checkout the Eval360 repository for our evaluation framework, it is a language model evaluation workspace built around the LM Evaluation Harness. It provides opinionated scripts and automation for benchmarking large checkpoints on reasoning, math, and code suites while coordinating large-cluster workflows (SLURM, Ray, and vLLM). The repository glues together local checkpoints, Hugging Face models, and multi-node serving endpoints to streamline end-to-end evaluation runs.

# Citation
```
@misc{k2team2025k2v2360openreasoningenhancedllm,
      title={K2-V2: A 360-Open, Reasoning-Enhanced LLM}, 
      author={K2 Team and Zhengzhong Liu and Liping Tang and Linghao Jin and Haonan Li and Nikhil Ranjan and Desai Fan and Shaurya Rohatgi and Richard Fan and Omkar Pangarkar and Huijuan Wang and Zhoujun Cheng and Suqi Sun and Seungwook Han and Bowen Tan and Gurpreet Gosal and Xudong Han and Varad Pimpalkhute and Shibo Hao and Ming Shan Hee and Joel Hestness and Haolong Jia and Liqun Ma and Aaryamonvikram Singh and Daria Soboleva and Natalia Vassilieva and Renxi Wang and Yingquan Wu and Yuekai Sun and Taylor Killian and Alexander Moreno and John Maggs and Hector Ren and Guowei He and Hongyi Wang and Xuezhe Ma and Yuqi Wang and Mikhail Yurochkin and Eric P. Xing},
      year={2025},
      eprint={2512.06201},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.06201}, 
}

```

