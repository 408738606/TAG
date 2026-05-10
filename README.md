# TAG: A Simple Yet Effective Temporal-Aware Approach for Zero-Shot Video Temporal Grounding
This repository contains the implementation of our BMVC 2025 paper, 
["TAG: A Simple Yet Effective Temporal-Aware Approach for Zero-Shot Video Temporal Grounding"](https://arxiv.org/pdf/2508.07925).

TAG is a simple yet effective temporal-aware framework for zero-shot Video Temporal Grounding (VTG).  
By leveraging temporal pooling, temporal coherence clustering, and similarity adjustment, TAG captures contextual continuity in videos and mitigates **segment fragmentation**, where semantically consistent frames are mistakenly split across multiple segments.  
This enables accurate localization of target moments without any training or reliance on large language models.  
TAG achieves state-of-the-art performance on Charades-STA and ActivityNet Captions, demonstrating strong generalization and robustness across various settings.
## Method Overview

<div align="center">
  <img src="figures/overview.png" alt="Method Overview" width="90%">
</div>

## Pipeline

<div align="center">
  <img src="figures/pipeline.png" alt="Pipeline" width="80%">
</div>

## 新增创新点（中文）

基于项目内论文（TAG、TimeXL、Moment-GPT、FreqCycle）的核心思想，对 TAG 进行了以下增强：

1. **频率感知的多尺度时序平滑**：利用相似度序列的频域能量分布，动态加权短/中/长窗口的时序平滑，增强对不同节奏与周期的动作片段建模能力（借鉴 FreqCycle 的多尺度时频建模思路）。
2. **反思式动态一致性重打分**：引入动态变化信号作为“反思”反馈，对候选片段的静态相似度进行一致性再加权，缓解碎片化边界与过度偏置（借鉴 TimeXL 的“预测-反思-修正”闭环思想）。
3. **查询去偏与扩展**：对输入查询做规则化去噪与轻量同义改写，并将多版本查询的候选片段进行融合，以降低语言偏差带来的定位误差（借鉴 Moment-GPT 的查询去偏思想，可通过 `--query_refine` 启用）。
4. **长度先验的跨度选择**：根据查询长度引入跨度长度先验，对过短或过长的候选跨度进行软约束，提升跨度分布合理性（借鉴 Moment-GPT 的跨度分布调节思路）。

### 新颖性检索说明

- 已对仓库内提供的论文进行对比，未发现与上述组合式改进完全相同的方案描述。
- 尝试进行线上关键词检索（arXiv 等）以确认是否已有相同发表，但当前环境网络访问受限无法完成外部检索。若需更严格的新颖性确认，建议在本地补充检索与查重。


## Quick Start

### Requiments
- python=3.8
- pytorch==2.0.1
- torchvision==0.15.2
- pytorch-cuda=11.7
- torchaudio==2.0.2
- tqdm
- scikit-learn
- ipykernel
- seaborn
- statsmodels
- patsy
- salesforce-lavis


## Data Preparation

Before training or evaluation, please prepare the datasets and extract visual features as follows.

### 1. Download Datasets

#### Charades-STA
- Download the Charades video dataset from the [Charades Project Page](https://prior.allenai.org/projects/charades).
- Extract the videos into your desired directory, e.g.: `videos/Charades/`


#### ActivityNet Captions
- Visit the [ActivityNet Download Page](http://activity-net.org/download.html) and request access to the dataset.
- Once approved, download the video dataset from ActivityNet.
- Extract the videos into your desired directory, e.g.: `videos/ActivityNet/`


### 2. Extract Visual Features

We use the BLIP-2 image-text matching model (pretrained on COCO) to extract visual features at **3 FPS**.

For Charades-STA:
```bash
python feature_extraction.py \
  --input_root videos/Charades/ \
  --save_root datasets/Charades/
```

For ActivityNet:
```bash
python feature_extraction.py \
  --input_root videos/ActivityNet/ \
  --save_root datasets/ActivityNet/
```

## Main Results

### Standard Split

```bash
# Charades-STA dataset
python evaluate.py --dataset charades --llm_output dataset/charades-sta/llm_outputs.json --tckmeans

# ActivityNet dataset
python evaluate.py --dataset activitynet --llm_output dataset/activitynet/llm_outputs.json --tckmeans
```

| Dataset        | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----         | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA  |  67.82  |  48.58  |  26.67  |  45.69  |
|  ActivityNet   |  51.88  |  28.91  |  15.07  |  36.55  |


### OOD Splits

```bash
# Charades-STA OOD-1
python evaluate.py --dataset charades --split OOD-1 --tckmeans

# Charades-STA OOD-2
python evaluate.py --dataset charades --split OOD-2 --tckmeans

# ActivityNet OOD-1
python evaluate.py --dataset activitynet --split OOD-1 --tckmeans

# ActivityNet OOD-2
python evaluate.py --dataset activitynet --split OOD-2 --tckmeans
```

| Dataset              | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----               | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA OOD-1  |  68.25  |  45.27  |  23.20  |  44.71  |
|  Charades-STA OOD-2  |  68.31  |  44.11  |  21.99  |  44.62  |
|  ActivityNet OOD-1   |  51.42  |  28.52  |  14.68  |  36.19  |
|  ActivityNet OOD-2   |  51.23  |  28.34  |  14.54  |  36.05  |


```bash
# Charades-CD test-ood
python evaluate.py --dataset charades --split test-ood --tckmeans

# Charades-CG novel-composition
python evaluate.py --dataset charades --split novel-composition --tckmeans

# Charades-CG novel-word
python evaluate.py --dataset charades --split novel-word --tckmeans
```

| Dataset                           | IoU=0.3 | IoU=0.5 | IoU=0.7 |  mIoU   |
| :-----                            | :-----: | :-----: | :-----: | :-----: |
|  Charades-STA test-ood            |  68.09  |  49.45  |  26.58  |  45.97  |
|  Charades-STA novel-composition   |  64.38  |  43.55  |  21.30  |  41.95  |
|  Charades-STA novel-word          |  68.49  |  52.37  |  32.66  |  47.86  |


## Acknowledgement

This repository is built upon the official implementation of [TFVTG (ECCV 2024)](https://github.com/minghangz/TFVTG). We thank the authors for their valuable contributions and open-source code.
