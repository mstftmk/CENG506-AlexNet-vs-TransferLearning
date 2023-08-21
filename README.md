# AlexNet Implementation and Transfer Learning Comparison

![Alexnet](https://learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png)
> AlexNet Architecture

This assignment provided in the CENG506 Deep Learning course focuses on implementing the AlexNet architecture from scratch and utilizing transfer learning to compare its performance. The primary goal is to evaluate the effectiveness of transfer learning in improving the accuracy of the model. The project also incorporates L2 regularization to reduce oscillations during training.

## Dataset

The dataset is include 4 different animal images. Dataset has 4 classes(Bear, Elephant, Leopard and Zebra). Each class has 350 images.

Ensure that you have the unzip the dataset and ready before running the code. Please follow the instructions provided within the code to load and preprocess the dataset.

## Requirements

Make sure you have the following dependencies installed before running the code:

```python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random

from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torch.optim as optim
```

## Usage

1. Clone the repository to your local machine using the following command:

```shell
git clone https://github.com/mstftmk/CENG506-AlexNet-vs-TransferLearning.git
```

2. Navigate to the project directory:
```shell
cd repo-path
```

3. Launch Jupyter Notebook:

```shell
jupyter notebook
```

4. Open the **Alexnet_vs_TransferLearning.ipynb** file from the Jupyter Notebook interface.

5. Follow the instructions provided within the notebook to run the code cells and execute the project.

The notebook will start training the AlexNet model from scratch and output the accuracy values after 30 epochs. Additionally, it will train the transfer learning model and display the accuracy achieved in 10 epochs. The comparison between the two models will be presented, highlighting the potential improvement achieved using transfer learning.

## Results

After completing the training process, the project will provide the following results:

- Accuracy value of the model built from scratch after 30 epochs: **%66**
- Accuracy value of the transfer learning model after 10 epochs: **%94**
- Comparison between the two models: **_%28_**

## Contributions

Contributions to the project are welcome! If you find any bugs or want to enhance the functionality, feel free to open an issue or submit a pull request.
