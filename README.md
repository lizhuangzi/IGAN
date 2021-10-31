# Information-Growth Attention Network for Image Super-Resolution

***** *I am now clearing up my code, and I will publish code within the next few days.**

This is the code of ACM Multimedia 2021 Paper: Information-Growth Attention Network for Image Super-Resolution

Abstract:
It is widely believed that a high-resolution (HR) image contains more productive information compared with its low-resolution (LR) versions, so image super-resolution (SR) satisfies an information-growth process. Considering the property, we attempt to exploit the growing information via a particular attention mechanism. In this paper, we propose a concise but effective Information-Growth Attention Network (IGAN) that shows the incremental information is beneficial for SR. Specifically, a novel information-growth attention mechanism is proposed. It aims to pay attention to features involving large information-growth capacity by assimilating the difference from current features to the former features. We also illustrate its effectiveness contrasted by widely-used self-attention mechanism using entropy and generalization bound analysis. Besides, existing channel-wise attention generation modules (CAGMs) have large information attenuation due to directly calculating global mean for feature maps. To solve it, we present an innovative CAGM that progressively decreases feature maps' sizes. Extensive experiments conducted on publicly available datasets demonstrate IGAN outperforms state-of-the-art attention-aware SR approaches.

## Requirements



## Pretrained Model

4x Model:  https://drive.google.com/file/d/12IKFNnjtfRauoR3hcuiq9OAxJvQsH0Bb/view?usp=sharing

8x Model: https://drive.google.com/file/d/19u1O1vtzql4zx78EtVgB9XSBj4vj3zyA/view?usp=sharing

## Dataset

We use the Div2k dataset for training. its webpage is available at: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

* Size of Dataset: Total 900 2k images about 7.12 GB
* Training: 800 images
* Testing: 100 images
* Benchmark dataset:
* Data format: .png
* PS: Data will be processed at ./src/data/Div2k.py


## Cite

The references introduced by .bib can be written as:

```
@inproceedings{Liz2021,
  title={Information-Growth Attention Network for Image Super-Resolution.},
  author={Zhuangzi Li, Ge Li, Thomas Li, Shan Liu and Wei Gao},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={544--552},
  year={2021}
}
```
