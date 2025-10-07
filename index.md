---
title: Hybrid Tile Sparsity
layout: home
nav_order: 0
---

<img src="pages/media/hybrid_sparsity_logo.svg" alt="Hybrid Tile Sparsity Logo" style="width:300px;"/>
<div class="btn-row" style="text-align:center;">
  <a href="https://github.com/Paramathic/patch" class="btn btn-blue">PATCH GitHub</a>
  <a href="https://github.com/Paramathic/stoicc" class="btn btn-purple">STOICC GitHub</a>
</div>

# **Introduction**

Although large language models (LLMs) excel in understanding and generating natural language, their enormous parameter sizes make them costly to operate, imposing substantial memory demands and high inference expenses, particularly during deployment. To address these challenges, techniques such as quantization and pruning have emerged to reduce inference costs while aiming to preserve model accuracy, though often with trade-offs compared to their denser counterparts.


Quantization effectively lowers the memory and computational overhead of LLMs, but it encounters significant hurdles in low-bitwidth scenarios.[^1]<sup>,</sup>[^12] Recent research has demonstrated that integrating sparsity with quantization offers a promising alternative to further advance LLM compression.[^2]<sup>,</sup>[^3]<sup>,</sup>[^13] However, the choice of sparsity format critically influences both the model's accuracy and its inference speed, necessitating a careful balance between these factors.

Unstructured sparsity, where non-zero elements can appear anywhere in a weight matrix, enables models to maintain high accuracy even when up to 50% of weights are pruned. Methods like SparseGPT[^3] and Wanda[^4] facilitate such pruning with minimal performance degradation. However, while unstructured sparsity provides strong compression benefits, it is challenging to accelerate on modern GPUs due to irregular memory access patterns. Hardware-optimized approaches, such as FlashLLM[^5], typically achieve meaningful inference speedups only at extreme sparsity levels (80% or higher). This tension between accuracy retention and hardware efficiency highlights the value of semi-structured sparsity formats, like the 2:4 pattern, which strike a more practical balance between performance and deployability.

Semi-structured sparsity patterns, including the 2:4 format[^6] supported by NVIDIA and AMD GPUs, deliver tangible speedups for large-scale model inference. Unlike the flexibility of unstructured sparsity, however, the 2:4 pattern imposes strict constraints by requiring exactly two of every four consecutive elements to be zero. This rigidity frequently results in notable accuracy drops when applied via one-shot pruning methods.[^3]<sup>,</sup>[^4]<sup>,</sup>[^7] Furthermore, studies indicate that sparsity should be distributed adaptively across layers for optimal results, rather than uniformly as enforced by 2:4.[^8]<sup>,</sup>[^9]<sup>,</sup>[^10] These drawbacks reveal that 2:4 sparsity alone falls short, emphasizing the necessity for hybrid strategies that combine the best of both worlds.

# **Hybrid Tile Sparsity**

To overcome these limitations, we introduce two complementary innovations: PATCH and STOICC. 
- [PATCH]({% link pages/patch.md %}): PATCH learns a hybrid mask that divides each weight matrix into hardware-friendly tiles, classifying each tile as either fully dense (0% sparsity) or 2:4 sparse (50% sparsity). This adaptive masking enables the matrix to achieve an effective global sparsity ratio between 0% and 50%, preserving accuracy in sensitive regions while applying efficient sparsity elsewhere. 
- [STOICC]({% link pages/stoicc.md %}): Complementing this, the STOICC compiler—built atop OpenAI's Triton[^11], seamlessly accelerates PATCH-generated models through its robust support for hybrid sparsity.

When combining STOICC and PATCH on LLaMA-2 7B deployed on a consumer-grade A6000 GPU, we realize 1.18×–1.38× end-to-end speedups over the dense baseline, alongside accuracy gains of 0.37%–2.96% relative to the leading 2:4 pruning method, MaskLLM.


### References
[^1]: [Lin, J., et al. AWQ: Activation-aware weight quantization for on-device llm compression and acceleration. MLSys, 2024.](https://arxiv.org/abs/2306.00978)
[^2]: [Mozaffari, M., et al. SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs. ICML 2025.](https://arxiv.org/abs/2410.09615)
[^3]: [Frantar, E., et al. Sparsegpt: Massive language models can be accurately pruned in one-shot. ICML 2023.](https://arxiv.org/abs/2301.00774)
[^4]: [Sun, M., et al. A simple and effective pruning approach for large language models. ICLR 2024.](https://arxiv.org/abs/2306.11695)
[^5]: [Xia, H., et al. Flash-llm: Enabling cost-effective and highly-efficient large generative model inference with unstructured sparsity.](https://arxiv.org/abs/2309.10285)
[^6]: [Mishra, A., et al. Accelerating sparse deep neural networks.](https://arxiv.org/abs/2104.08378)
[^7]: [Fang, G., et al. MaskLLM: Learnable semistructured sparsity for large language models..](https://arxiv.org/abs/2409.17481)
[^8]: [Yin, L., et al. Outlier weighed layerwise sparsity (OWL): A missing secret sauce for pruning llms to high sparsity, 2025.](https://arxiv.org/abs/2310.05175)
[^9]: [Wang, W., et al. Rethinking the value of transformer components, 2020.](https://arxiv.org/abs/2011.03803)
[^10]: [Lee, J., et al. Layer-adaptive sparsity for the magnitudebased pruning, 2021.](https://arxiv.org/abs/2010.07611)
[^11]: [OpenAI Triton](https://triton-lang.org/)
[^12]: [Frantar, E., et al. Optq: Accurate quantization for generative pre-trained transformers. ICLR, 2022.](https://arxiv.org/abs/2210.17323)
[^13]: [Mozaffari, M., et al. When Quantization Isn’t Enough: Why 2:4 Sparsity Matters](https://pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters/)