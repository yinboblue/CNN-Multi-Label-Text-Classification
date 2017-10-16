# Convolutional Neural Networks for Multi-Label Text Classification

This project is my research group project, and it is also a study of TensorFlow, Deep Learning(CNN, RNN, LSTM, etc.).

The main objective of the project is to solve the multi-label text classification problem based on Convolutional Neural Networks. Thus, the format of the data label is like [0, 1, 0, ..., 1, 1] according to the characteristics of such problem.

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

Embedding Layer + Convolutional Layer + Max-Pooling Layer + FC Layer + Softmax

![](https://farm5.staticflickr.com/4500/36676786824_2a45d53de3_o.png)

## Innovation

1. Make the data support **Chinese** and English.(Which use `gensim` seems easy)
2. Can use **your own pre-trained word vectors**.
3. Add a new **Highway Layer**.
4. Add **parent label bind** to limit the output of the prediction label.
5. Can choose **train** the model directly or **restore** the model from checkpoint.  
6. Add **model test code**. 


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
