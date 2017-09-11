# Convolutional Neural Networks for Multi-Label Text Classification

This project is used by my bachelor graduation project, and it is also a study of TensorFlow, familiar with CNN, RNN and other neural networks.

The main objective of the project is to determine whether the two sentences are similar in sentence meaning (binary classification problems) by the two given sentences.

The project refer to [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf), make the data helper supports Chinese language (Task required) and modified the network structure (Based on my task).

## Requirements

- Python 3.x
- Tensorflow 1.0.0 +
- Numpy
- Gensim

## Data

Research data may attract copyright protection under China law. Thus, there is only code.

实验数据属于实验室与某公司的合作项目，涉及商业机密，在此不予提供，还望谅解。

## Pre-trained Word Vectors

Use `gensim` package to pre-train my data.


## Network Structure

![](https://farm1.staticflickr.com/650/33049175050_080d4de7ff_o.jpg)

## Innovation

1. Make the data support Chinese and English.(Which use `gensim` seems easy)
2. Can use your own pre-trained word vectors.
3. Deign two subnetworks to meet the task requirements.
4. Add a new Highway Layer.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
