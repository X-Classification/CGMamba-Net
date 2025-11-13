# CGMamba-Net
<div align="center">
<h1>CGMamba-Net</h1>
<h3>Enhancing Skin Lesion Classification with Concentric Group Mamba Architecture</h3>
<div>

## Abstract

Skin lesion classification presents a significant challenge in dermatological diagnostics due to substantial intra-class variations and inter-class similarities. Conventional CNNs, while effective in capturing local textures, often fall short in modeling spatial dependencies between lesion interiors and edges. Addressing this gap, we introduce CGMamba-Net, a novel network integrating an improved Bidirectional Concentric Group Mamba (BCGMamba) block into the ConvNeXt framework. The BCGMamba block employs four complementary concentric scanning strategies to explicitly model lesion center-edge relationships, enhancing sensitivity to boundary irregularities and textures. Experiments on the HAM10000 and ISIC2019 datasets demonstrate the superiority of CGMamba-Net, achieving classification accuracies of 95.20% and 93.20%, respectively. Our findings highlight the potential of combining local and global feature modeling for accurate skin lesion classification.

## Overview

<div align="center">
<img src="assets/CGMamba-Net.png" />
</div>

## Getting Started

### Installation

**Step 1: Clone the CGMamba repository:**

To get started, first clone the CGMamba repository and navigate to the project directory:

```bash
git clone https://github.com/X-Classification/CGMamba-Net.git
cd CGMamba
```

**Step 2: Environment Setup:**

We recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n cgMamba
conda activate cgMamba
```


***Install Dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```
<!-- cd kernels/cross_scan && pip install . -->



### Dataset Preparation
Download the [ISIC2019](https://challenge.isic-archive.com/data/) classification dataset and structure the data as follows:
```
/path/to/ISIC2019/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
### Model Training

To train CGMamba models for classification on your dataset, use the following commands for different configurations:

```bash
python main.py
```