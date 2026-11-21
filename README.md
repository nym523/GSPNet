
<div align="center">

# GSPNet

### Geometric-Statistical Joint-Prior Network for 3-D Change Detection


[Paper](#citation) | [Code](#code-release) | [Results](#experimental-results)

</div>

## Overview

This repository is dedicated to **GSPNet**, a Geometric-Statistical Joint-Prior Network for 3-D Change Detection.

GSPNet integrates uncertainty-aware geometric priors, latent multi-modal statistical priors, and balanced supervision to improve change detection performance in complex 3-D scenes.

Our method addresses three major challenges:

1. **Limited geometric discrimination:** Reliance on inter-point distances and fixed thresholds can overlook directional changes and lead to pseudo-change responses.

2. **Insufficient modeling of intra-class variations:** Discrete semantic labels cannot fully characterize different latent feature states within the same class.

3. **Optimization bias under class imbalance:** Frequent classes may dominate training, limiting the learning of rare change categories.

## Framework

![GSPNet Architecture](assets/GSPNet_framework.png)

The proposed GSPNet adopts a Siamese encoder–decoder architecture for bi-temporal point cloud change detection.

## Main Components

### Uncertainty-Aware Geometric Prior (UAGP)

UAGP jointly models positional, absolute-orientation, and relative-orientation differences. Second-order graph propagation aggregates local geometric information, while surface roughness and curvature are employed to construct adaptive detection limits for pseudo-change suppression.

### Latent Multi-Modal Statistical Prior (LMSP)

LMSP constructs cross-temporal change descriptors from deep features. A Gaussian Mixture Model (GMM) characterizes latent sub-modes within the Changed and Unchanged classes. Bayesian posterior inference then estimates point-level change probabilities.

### Auxiliary Balanced Segmentation Head (ABSH)

ABSH preserves full-sample supervision through the main segmentation head and introduces an auxiliary branch with class-dependent random retention. This produces balanced sample counts in expectation and mitigates optimization bias caused by class imbalance.

## Experimental Results

| Dataset | mAcc (%) ↑ | mIoUch (%) ↑ |
|:---|---:|---:|
| Urb3DCD-V2 | **95.93** | **88.56** |
| SLPCCD | **79.66** | **62.39** |
| HKCD | **88.32** | **79.38** |

## Datasets

Experiments are conducted on three public datasets:

- **Urb3DCD-V2**
- **SLPCCD**
- **HKCD**

## Citation

If you find this work useful, please consider citing our paper.

**Paper:** GSPNet: Geometric-Statistical Joint-Prior Network for 3-D Change Detection


