# CNNs meet ViTs

This repository include an unofficial implementation of the CMT architecture introduced in the [CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/abs/2107.06263) paper by Guo et al. (2021).

Each `.ipynb` notebook defines a total of four models:

| Model | Architecture | Description |
|:-:|:-:|:--|
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_1) | CMT | CMT-Ti with lightweight multi-head self-attention |
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_2) | CMT | CMT-Ti with standard multi-head self-attention |
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_3) | CMT | CMT-Ti without multi-head self-attention |
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_4) | ResNet | ResNet-18 (224 × 224) or ResNet-110 (32 × 32) |

The definitions of all modules needed to define the CMT-Ti architecture can be found in the `CMT.py` file.

## Training

All models are trained for ![](https://latex.codecogs.com/gif.latex?%5Cinline%2025) epochs, using cross-entopy, AdamW with an amount of weight decay equals to ![](https://latex.codecogs.com/gif.latex?%5Cinline%201e%5Ctext%7B-%7D5), an initial learning rate of ![](https://latex.codecogs.com/gif.latex?%5Cinline%206e%5Ctext%7B-%7D5), and a cosine annealing learning rate schedule. 

## Results

- `CNNs_meet_ViTs_224x224.ipynb`:

![](https://i.ibb.co/RQVhStS/download.png)
![](https://i.ibb.co/VLPFsNr/download-1.png)

- `CNNs_meet_ViTs_32x32.ipynb`:

![](https://i.ibb.co/BTWQtrS/download.png)
![](https://i.ibb.co/WD14jc8/download-1.png)
