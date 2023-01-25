# CNNs meet ViTs

This repository include an unofficial implementation of the CMT architecture introduced in the [CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/abs/2107.06263) paper by Guo et al. (2021).

The `.ipynb` notebook includes the definition of all modules needed to define the CMT-Ti architecture with the only modification of having an input resolution of ![](https://latex.codecogs.com/gif.latex?%5Cinline%20224%20%5Ctimes%20224) instead of ![](https://latex.codecogs.com/gif.latex?%5Cinline%20160%20%5Ctimes%20160). Moreover, the aforementioned notebook defines and train  the following four different models on CIFAR-10:

| Model | Architecture | Description | #Parameters | #FLOPs | Accuracy |
|:-:|:-:|:--|:-:|:-:|:-:|
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_1) | CMT | CMT-Ti with lightweight multi-head self-attention | 9.01M | 1.31B | 88.79% |
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_2) | CMT | CMT-Ti with standard multi-head self-attention | 8.11M | 3.56B | 88.82% |
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_3) | CMT | CMT-Ti without multi-head self-attention | 5.58M | 0.95B | 89.05% |
| ![](https://latex.codecogs.com/gif.latex?%5Cinline%20m_4) | ResNet | ResNet-18 | 11.69M | 1.83B | 87.67%|

All models have been trained for ![](https://latex.codecogs.com/gif.latex?%5Cinline%2025) epochs, using cross-entopy, AdamW with an amount of weight decay equals to ![](https://latex.codecogs.com/gif.latex?%5Cinline%201e%5Ctext%7B-%7D5), an initial learning rate of ![](https://latex.codecogs.com/gif.latex?%5Cinline%206e%5Ctext%7B-%7D5), and a cosine annealing learning rate schedule.

## Results

The following plot shows the validation loss and accuracy during training:

![](https://i.ibb.co/nqXMhNk/download.png)

while, the following one compares the performances of all models with respect to the total number of FLOPs:

![](https://i.ibb.co/WsQppY1/download-1.png)
