<div align="center">
    <h1>NaLA</h1>
    <h3>Natural Analysis of Language Attitudes in BlueSky Conversations</h3>
</div>

# Introduction

In recent years, sentiment analysis on social media posts has emerged as a key tool for understanding collective mood, 
the diffusion of opinions and public perception in real time. This particular type of analysis has proven useful in a wide 
range of contexts, including monitoring reactions to political events, communication campaigns, health crises and marketing campaigns. 

In this context, the present work aims to explore and compare different sentiment analysis approaches, while withstanding increasing levels of modeling and computational complexity. The analysis is conducted on a subset of texts extracted from posts from the social media [Bluesky](https://bsky.app), an emerging platform that is rapidly gaining popularity as a decentralized alternative to traditional social networks. 

The goal is to evaluate the performance of the models with respect to computational constraints, accuracy and simplicity of implementation, with particular attention to the trade-off between classification quality and computational cost. In particular, the following will be compared:
- Classifical _machine learning_ methods, such as: Random Forest and Naive Bayes;
- _Deep Learning_ methods, such as: MLP, bidirectional RNN, BERTweet and RoBERTa.

# Technical Report

To know more about the technical details of this work, you can access the technical report under `deliverables` folder.
At the moment, only the italian version is available. An english version is under development and will be updated soon.

# Dataset

The original dataset employed in this study is available at [withalim/bluesky-posts](https://huggingface.co/datasets/withalim/bluesky-posts).

The extracted and processed subset employed for training and validation is available at the following link: [here](https://drive.google.com/drive/folders/1xcUxIP4C3Mhh7R4exW4TsWHK6R80tCty?usp=sharing).

# Results

Our findings are somewhat unexpected: simpler models, such as MLP and Naive Bayes, outperformed more complex architectures, including RNNs and pre-trained models like BERTweet and RoBERTa.

This underscores the principle that model complexity does not always translate to better performance. 
This is particularly important in resource-constrained environments, where efficiency and simplicity can offer significant advantages.

The results are summarized in the table below.

| **Model**        | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Training time** | **# Parametri** |
|------------------|--------------|---------------|------------|--------------|-------------------|------------------|
| RandomForest     | 0.60         | 0.58          | 0.59       | 0.60         | 4 hours           | 1746             |
| **MLPClassifier** | **0.65**     | **0.65**      | **0.66**   | **0.65**     | 2 hours           | **1101**             |
| Naive Bayes      | 0.63         | 0.63          | 0.64       | 0.63         | 1 minute          | 835K             |
| RNN              | 0.38         | 0.38          | 0.37       | 0.37         | 15 hours          | 16M              |
| BERTweet         | 0.59         | 0.60          | 0.59       | 0.59         | -                 | 134M             |
| RoBERTa          | 0.59         | 0.60          | 0.59       | 0.59         | -                 | 124M             |

In the following figure is presented a more direct visualization of the difference between models size and performance.



# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.10` or higher.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).
## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/amigli/NaLA.git
```

## Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```