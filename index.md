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


Quantization effectively lowers the memory and computational overhead of LLMs, but it encounters significant hurdles in low-bitwidth scenarios.[^1]<sup>,</sup>[^2] Recent research has demonstrated that integrating sparsity with quantization offers a promising alternative to further advance LLM compression.[^3][^12]

Unstructured sparsity, where non-zero elements can appear anywhere in a weight matrix, enables models to maintain high accuracy even when up to 50% of weights are pruned. Methods like SparseGPT[^3] and Wanda[^4] facilitate such pruning with minimal performance degradation. However, while unstructured sparsity provides strong compression benefits, it is challenging to accelerate on modern GPUs due to irregular memory access patterns. Hardware-optimized approaches, such as FlashLLM[^5], typically achieve meaningful inference speedups only at extreme sparsity levels (80% or higher). This tension between accuracy retention and hardware efficiency highlights the value of semi-structured sparsity formats, like the 2:4 pattern, which strike a more practical balance between performance and deployability.

Semi-structured sparsity patterns, including the 2:4 format[^6] supported by NVIDIA and AMD GPUs, deliver tangible speedups for large-scale model inference. Unlike the flexibility of unstructured sparsity, however, the 2:4 pattern imposes strict constraints by requiring exactly two of every four consecutive elements to be zero. This rigidity frequently results in notable accuracy drops when applied via one-shot pruning methods.[^3]<sup>,</sup>[^4]<sup>,</sup>[^7] Furthermore, studies indicate that sparsity should be distributed adaptively across layers for optimal results, rather than uniformly as enforced by 2:4.[^8]<sup>,</sup>[^9]<sup>,</sup>[^10] These drawbacks reveal that 2:4 sparsity alone falls short, emphasizing the necessity for hybrid strategies that combine the best of both worlds.

# **Hybrid Tile Sparsity**

To overcome these limitations, we introduce two complementary innovations: PATCH and STOICC. 
- [PATCH](./pages/patch.md): PATCH learns a hybrid mask that divides each weight matrix into hardware-friendly tiles, classifying each tile as either fully dense (0% sparsity) or 2:4 sparse (50% sparsity). This adaptive masking enables the matrix to achieve an effective global sparsity ratio between 0% and 50%, preserving accuracy in sensitive regions while applying efficient sparsity elsewhere. 
- [STOICC](./pages/stoicc.md): Complementing this, the STOICC compiler—built atop OpenAI's Triton[^11], seamlessly accelerates PATCH-generated models through its robust support for hybrid sparsity.

When combining STOICC and PATCH on LLaMA-2 7B deployed on a consumer-grade A6000 GPU, we realize 1.18×–1.38× end-to-end speedups over the dense baseline, alongside accuracy gains of 0.37%–2.96% relative to the leading 2:4 pruning method, MaskLLM.


### References
[^1]: [Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., and Han, S. AWQ: Activation-aware weight quantization for on-device llm compression and acceleration. MLSys, 2024.](https://arxiv.org/abs/2306.00978)
[^2]: [Mohammad Mozaffari, Amir Yazdanbakhsh, and Maryam Mehri Dehnavi. SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs, 2025a.](https://arxiv.org/abs/2410.09615)
[^3]: [Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In ICML, 2023.](https://arxiv.org/abs/2301.00774)
[^4]: [Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models.](https://arxiv.org/abs/2306.11695)
[^5]: [Xia, H., Zheng, Z., Li, Y., Zhuang, D., Zhou, Z., Qiu, X., Li, Y., Lin, W., and Song, S. L. Flash-llm: Enabling cost-effective and highly-efficient large generative model inference with unstructured sparsity.](https://arxiv.org/abs/2309.10285)
[^6]: [Mishra, A., Latorre, J. A., Pool, J., Stosic, D., Stosic, D., Venkatesh, G., Yu, C., and Micikevicius, P. Accelerating sparse deep neural networks.](https://arxiv.org/abs/2104.08378)
[^7]: [Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023.](https://arxiv.org/abs/2409.17481)
[^8]: [Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, et al. Outlier weighed layerwise sparsity (OWL): A missing secret sauce for pruning llms to high sparsity, 2025.](https://arxiv.org/abs/2310.05175)
[^9]: [Wenxuan Wang and Zhaopeng Tu. Rethinking the value of transformer components, 2020.](https://arxiv.org/abs/2011.03803)
[^10]: [Jaeho Lee, Sejun Park, Sangwoo Mo, Sungsoo Ahn, et al. Layer-adaptive sparsity for the magnitudebased pruning, 2021.](https://arxiv.org/abs/2010.07611)
[^11]: [OpenAI Triton](https://triton-lang.org/)
[^12]: [Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Optq: Accurate quantization for generative pre-trained transformers. In ICLR, 2022.](https://arxiv.org/abs/2210.17323)