# DF2RQ

## Introduction

Official code for 'DF2RQ: Dynamic Feature Fusion via Region-wise Queries for Semantic Segmentation of Multimodal Remote Sensing Data'

**News:**

* Our paper is still under review, the code will be uploaded gradually, and all detailed files will be released after paper acceptance.

## Installation

We have developed a training framework for remote sensing data post-spatial alignment supporting two or more modalities based on mmsegmentation, primarily including multi-modal dataset classes, data augmentation functions, and visualization tools.

Please follow the *[guideline](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)* to install the dev enviroments.

## Dataset Preparation

### Multimodal Semantic Segmentation Dataset for Remote Sensing

#### 4 Modality

**Globe230k**

```python
Globe230k
|-- dem_patch
|-- image_patch
|-- label_patch
|-- ndvi_patch
|-- vvvh_patch
|-- test_num.txt
|-- train_num.txt
`-- val_num.txt

```

#### 3 Modality

**C2seg**

```python
C2Seg
`-- BW
    |-- beijing
    |   |-- hsi
    |   |-- label
    |   |-- msi
    |   `-- sar
    `-- wuhan
        |-- hsi
        |-- label
        |-- msi
        `-- sar
```

#### 2 Modality

**Vaihigen and potsdam**

```
...

```

### Other Multimdodal Datatset

...

## Usage

### Multimodal Dataset Class

```python
mmseg/datasets/globe230k.py
...
```

### Multimodal Dataset Augmentation

```python
mmseg/datasets/transforms/transforms.py
mmseg/datasets/transforms/loading.py
mmseg/datasets/transforms/formating.py
```

### Multimodal Data Segmentation Visualization

```python
tools/mm_feature_map_visual_xxx.py
```
