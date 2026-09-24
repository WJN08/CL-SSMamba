# CL-SSMamba

PyTorch implementation of **CL-SSMamba* , a continual learning framework for hyperspectral anomaly detection (HAD) based on collaborative spatial-spectral Mamba learners.

## Introduction

Hyperspectral anomaly detection in continuously changing scenarios faces two major challenges: catastrophic forgetting during sequential learning and insufficient spatial-spectral representation of complex hyperspectral scenes.

CL-SSMamba addresses these challenges through a collaborative multi-learner architecture. Multiple parallel learners jointly reconstruct the dominant background while retaining model parameters from previous tasks instead of replaying historical hyperspectral samples.

Each learner combines:

- Global spatial Mamba modeling for long-range spatial dependencies;
- Local spectral Mamba modeling for cross-band spectral correlations;
- Spectral correlation enhancement based on orthogonal feature projections;
- Background reconstruction for residual-based anomaly detection.

During continual learning, reconstruction loss, ensemble cooperation loss, and historical parameter regularization are jointly optimized to balance plasticity and stability across sequential tasks.

## Framework

<p align="center">
  <img src="https://github.com/WJN08/CL-SSMamba/blob/main/Figure/CL-SSMamba.png?raw=true" width="70%">
</p>


The overall pipeline consists of:

1. Hyperspectral data preprocessing;
2. Parallel spatial-spectral Mamba learners;
3. Background reconstruction and learner fusion;
4. Continual parameter updating;
5. Reconstruction residual computation;
6. RX-based anomaly scoring.

For task \(t\), the reconstructed background is obtained by averaging the outputs of multiple parallel learners:
$$
\hat{X}_t =
\frac{1}{K}
\sum_{k=1}^{K}
f_{\theta_k^t}(\bar{X}_t).
$$
The reconstruction residual is subsequently calculated as
$$
R_t = X_t - \hat{X}_t,
$$
and the RX detector is applied to obtain the final anomaly map.
