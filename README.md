# Dog vs Cat Classifier

This project is a simple **Dog vs Cat image classification model** using **Convolutional Neural Networks (CNN)** implemented with **TensorFlow** and **Keras**. The model is trained on a dataset of dog and cat images and is used to classify an image as either a dog or a cat.

[Watch the video here](https://github.com/karan89200/DOG_VS_CAT/blob/main/ezyZip.mp4)


## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [How to Use](#how-to-use)
5. [Model Details](#model-details)
6. [Video Explanation](#video-explanation)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview

In this project, a machine learning model is built to classify images of dogs and cats. The dataset used for training contains labeled images of both dogs and cats. The model is built using a Convolutional Neural Network (CNN), which is particularly effective for image classification tasks.

The application allows users to upload an image and get a prediction of whether the image contains a dog or a cat. 

### Key Features:
- Image preprocessing to resize and normalize input images.
- CNN-based image classification model.
- Model trained on dog and cat images.
- Interactive web interface built with **Streamlit**.

## Requirements

Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- Pillow

You can install all the necessary dependencies using the `requirements.txt` file.

### `requirements.txt`
```txt
tensorflow==2.10.0
streamlit
opencv-python
numpy
pillow
