# $R^{2}$-dLLM: Accelerating Diffusion Large Language Models via Spatio-Temporal Redundancy Reduction

<a href='https://arxiv.org/abs/2604.18995'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href='https://huggingface.co/collections/ZhenbangDu/r2-dllm'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

The code for the paper [$R^{2}$-dLLM: Accelerating Diffusion Large Language Models via Spatio-Temporal Redundancy Reduction](https://arxiv.org/abs/2604.18995)
- Authors: Zhenbang Du, Kejing Xia, Xinrui Zhong, Yonggan Fu, Nicolai Oswald, Binfei Ji, Brucek Khailany, Pavlo Molchanov, and Yingyan (Celine) Lin.

## Abstract

Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to autoregressive generation by enabling parallel token prediction. However, practical dLLM decoding still suffers from high inference latency, which limits deployment. In this work, we observe that a substantial part of this inefficiency comes from recurring redundancy in the decoding process, including spatial redundancy caused by confidence clusters and positional ambiguity, and temporal redundancy caused by repeatedly remasking predictions that have already stabilized. Motivated by these patterns, we propose $R^{2}$-dLLM, a unified framework for reducing decoding redundancy from both inference and training perspectives. At inference time, we introduce training-free decoding rules that aggregate local confidence and token predictions, and finalize temporally stable tokens to avoid redundant decoding steps. We further propose a redundancy-aware supervised fine-tuning pipeline that aligns the model with efficient decoding trajectories and reduces reliance on manually tuned thresholds. Experiments demonstrate that $R^{2}$-dLLM consistently reduces the number of decoding steps by up to 88% compared to existing decoding strategies, while maintaining competitive generation quality across different models and tasks. These results validate that decoding redundancy is a central bottleneck in dLLMs, and that explicitly reducing it yields substantial practical efficiency gains.

<p align="center">
<img width="928" alt="image" src="figures/visualization.png"> 
</p>

## Citation

If you find this work useful for your research, please cite:

````
@article{du2026r,
  title={$R^{2}$-dLLM: Accelerating Diffusion Large Language Models via Spatio-Temporal Redundancy Reduction},
  author={Du, Zhenbang and Xia, Kejing and Zhong, Xinrui and Fu, Yonggan and Oswald, Nicolai and Ji, Binfei and Khailany, Brucek and Molchanov, Pavlo and Lin, Yingyan},
  journal={arXiv preprint arXiv:2604.18995},
  year={2026}
}
````
