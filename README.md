# Dual Context-Aware Negative Sampling Strategy for Graph-based Collaborative Filtering

[![Paper](https://img.shields.io/badge/Paper-CIKM%2725-blue)](https://link-to-your-paper.pdf)

Official PyTorch implementation of our CIKM 2025 paper:

> **Dual Context-Aware Negative Sampling Strategy for Graph-based Collaborative Filtering**  
> Xi Wu, Wenzhe Zhang, [Add other authors]  
> Accepted at *The 34th ACM International Conference on Information and Knowledge Management (CIKM 2025)*

---

## 📌 Introduction

Negative sampling plays a critical role in training graph-based collaborative filtering (GCF) models.  
Traditional negative sampling strategies often ignore **global** and **local** contextual information, leading to suboptimal performance.  

In this work, we propose **Dual Context-Aware Negative Sampling (DCANS)**, which integrates:
- **Global context**: semantic similarities across the entire user–item graph.
- **Local context**: neighborhood-aware sampling to capture fine-grained relationships.

Our method improves both **alignment** and **uniformity** in the learned embeddings, leading to more robust recommendations.

---

## 🚀 Features
- **Dual Context-Aware Sampling**: Combines global and local graph contexts for negative sample selection.
- **Plug-and-Play**: Can be easily integrated into existing GCF models such as LightGCN, NGCF, etc.
- **State-of-the-Art Performance**: Achieves significant improvement on multiple benchmark datasets.

---






## 📝 Citation
If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{wu2025dcans,
  title={Dual Context-Aware Negative Sampling Strategy for Graph-based Collaborative Filtering},
  author={Wu, Xi and Zhang, Wenzhe and Others},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year={2025},
  publisher={ACM}
}
```









