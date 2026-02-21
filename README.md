![mEoW-oR-wOoF](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/mEoW-oR-wOoF.jpg)

<br />
<p align="center">
  <h3 align="center">mEoW-oR-wOoF</h3>
  <p align="center">Logistic Regression Cat vs Dog Classifier</p>
</p>

## Introduction

**mEoW-oR-wOoF** is a machine learning project that classifies images as either **cats** or **dogs** using a logistic regression model.

The purpose of this project is to demonstrate a complete classical machine learning workflow for image classification, including:

- image preprocessing
- feature vector preparation
- model training
- evaluation metrics
- performance visualization

This project focuses on building a strong foundational understanding of machine learning pipelines using interpretable models rather than deep learning architectures.

## Getting Started


## Project Structure

- [`mEoW-oR-wOoF.jpg`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/mEoW-oR-wOoF.jpg) – Project image used for documentation or demonstration.
- [`pyproject.toml`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/pyproject.toml) – Project configuration and dependency definitions.
- [`README.md`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/README.md) – Main documentation and usage instructions.
- [`train.py`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/train.py) – Main training script for the logistic regression model.
- [`utils/`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/tree/main/utils) – Helper modules used throughout training and evaluation.
  - [`helper.py`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/utils/helper.py) – Utility functions for data handling and preprocessing.
  - [`lr_utils.py`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/utils/lr_utils.py) – Logistic regression helper functions.
  - [`__init__.py`](https://github.com/MosesTheRedSea/mEoW-oR-wOoF/blob/main/utils/__init__.py) – Package initializer.


## Installation

Clone the repo
 ```sh
 git clone https://github.com/MosesTheRedSea/mEoW-oR-wOoF.git
 ```
   
Setup Your Virtual Python Environment
 ```sh
 uv sync || uv run pyproject.toml
 ```

Activate The Virtual Python Environment
 ```sh
 source .venv/bin/activate
 ```

## Running Models

  ```sh
  python train.py
  ```
