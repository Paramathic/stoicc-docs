---
title: STOICC
layout: home
nav_order: 0
---

# **STOICC**: The **S**parse **T**ile **MO**sa**IC** **C**ompiler

## Overview

Sparsity is ubiquitous in domains such as machine learning, where weight matrices for linear layers can be pruned to create sparse matrices. To achieve the best balance between model accuracy, memory usage, and performance, different sparse formats are more optimal for different regions of the sparse matrix. However, current tools only allow a single sparse format to be used per matrix.

**We introduce STOICC, a novel tile-based sparsity compiler that enables multiple sparse formats to coexist in the same matrix.**

[Code Coming Soon](/){: .btn .btn-blue }

## Inspector-Executor Tools for Sparsity 

Compilers[^1] must optimize sparse matrix operations to achieve performance. This is commonly done in two phases:

* **The Inspector**: Inspects data sparsity pattern and performs reordering, compression, and scheduling.
* **The Executor**: Uses data and schedule to perform SpMM operations.

<img src="media/inspector_executor.svg"  alt="Inspector/Executor Framework">

## STOICC Pipeline

The STOICC [Inspector]({% link inspector_executor.md %}#inspector) takes in a sparse matrix. It then inspects the data sparsity to assign tile sparsity types, compress/reorder data, and create a schedule. This schedule is then used along with the data by an [Executor]({% link inspector_executor.md %}#executor) written in [Triton](https://triton-lang.org/) to perform the SpMM operation. Finally, the [Sparse Compiler]({% link sparse_compiler.md %})---our modified Triton compiler---is used to lower the executor code to the GPU. 

<img src="media/stoicc.svg"  alt="STOICC">

## Results

<!-- Results coming soon
{: .label .label-yellow } -->
We benchmark STOICC on an NVIDIA A100 (80GB) GPU with a mixture of dense tiles and [2:4](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) sparse tiles, where the 2:4 tiles are selected randomly. For example, "50% 2:4 Tiles" corresponds to a scenario where half of the tiles are pruned to 2:4 sparsity while the other half remain dense, resulting in an overall sparsity level of 25% for the whole matrix. The matrix sizes are chosen based on those used in the OPT family of models. 

<img src="media/heterogeneous_bs16.svg" alt="batch size 16 heterogeneous results">
<img src="media/heterogeneous_bs32.svg" alt="batch size 32 heterogeneous results">


We also compare STOICC performance with 100% 2:4 tiles against the CUTLASS 2:4 sparse kernel integrated in PyTorch.

<img src="media/full2_4_bs2048.svg" alt="2:4 comparison against CUTLASS 2:4, batch size 2k">
<img src="media/full2_4_bs4096.svg" alt="2:4 comparison against CUTLASS 2:4, batch size 4k">

----

### References

[^1]: [L. Wilkinson, K. Cheshmi, and M. M. Dehnavi, ‘Register Tiling for Unstructured Sparsity in Neural Network Inference’, Proc. ACM Program. Lang., vol. 7, no. PLDI, Jun. 2023.](https://doi.org/10.1145/3591302)