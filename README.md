# Brain-Tumour-Detection
Brain tumor detection and prediction using keras vgg-16

## Demo
  <b>Link:<b/> https://brain-tumor.netlify.app/
  <p>
  <img src="./Readme_resource/Image1.png" width="800" >
  <img src="./Readme_resource/Image2.png" width="800" >
  </p>
  
## Get Started

Create a new enviornment
```
conda create -p venv python==3.10.6 -y
```

Activate the enviornment

```
conda activate venv/
```

Install the requirements

```
pip install -r requirements.txt
```

# About the project

### Problem Statement:

The detection of brain tumors is a critical task in the field of medical imaging, as it can significantly impact patient outcomes. However, accurately detecting brain tumors can be challenging due to the complex nature of the human brain and the variability in the appearance of tumors. Therefore, there is a need for an automated and accurate brain tumor detection system that can assist healthcare professionals in diagnosing brain tumors.

The goal of this project is to develop a brain tumor detection system that can accurately classify brain scans as either containing a tumor or not. We aim to create a model that can handle the variability in brain scans and provide accurate results in a timely manner. Additionally, we aim to develop a user-friendly interface that can make our model accessible to healthcare professionals and patients alike.

To achieve our goal, we will use a dataset consisting of brain scan images with and without tumors. We will preprocess and augment the dataset to increase its size and variability. We will then use a pre-trained VGG16 model to train the data and generate predictions for new images.

Finally, we will create a web-based interface using React and Flask that allows users to upload and analyze brain scans using our model. By providing a user-friendly and accessible brain tumor detection system, we aim to improve the accuracy and speed of brain tumor diagnosis, ultimately leading to better patient outcomes.


## Machine learnng model

In this project, we aimed to develop a model that can accurately classify brain scans as either having a tumor or not. To achieve this, we used a dataset consisting of images of brain scans with and without tumors. However, since the dataset was relatively small, we augmented the data to increase its size and diversity.

Data augmentation involves generating additional images from the existing ones by applying various transformations such as rotation, flipping, scaling, and shifting. By doing so, we can create more samples that are similar to the original ones but have slightly different features, thereby increasing the variability in the data and making the model more robust.

After augmenting the data, we trained a VGG16 model on the dataset. The VGG16 is a deep convolutional neural network that is widely used for image classification tasks. The model consists of multiple convolutional and pooling layers followed by several fully connected layers. It can extract features from the input images and use them to classify the images into different categories.

During training, we used the augmented dataset to train the model using backpropagation, which involves adjusting the weights of the model based on the difference between the predicted and actual labels. We also used techniques such as dropout and regularization to prevent overfitting and improve the generalization of the model.

After training, we evaluated the performance of the model on a separate test set. We obtained an accuracy of 99% on the test set, which indicates that the model is able to accurately classify brain scans as either having a tumor or not.

In summary, by augmenting the dataset and training a VGG16 model, we were able to develop a highly accurate brain tumor detection system. This system has the potential to aid healthcare professionals in diagnosing brain tumors more quickly and accurately, which can ultimately improve patient outcomes.

Finally saved the model using pickle

### About VGG16 Model

The VGG16 model is a deep convolutional neural network that is widely used for image classification tasks. In the context of brain tumor detection, the VGG16 model can be trained to classify brain scans as either benign or malignant. Here is a brief description of how the VGG16 model works:

Input: The model takes in a brain scan as input. The input is typically a 3D array of pixel values representing the image.

Convolutional Layers: The input is passed through a series of convolutional layers, each of which extracts increasingly complex features from the input image. The convolutional layers use filters to scan the image and identify patterns and structures.

Pooling Layers: After each convolutional layer, the model uses pooling layers to reduce the spatial dimensions of the features while retaining their important information. This reduces the number of parameters in the model and helps prevent overfitting.

Flatten: Once the features have been extracted, the model flattens them into a 1D vector.

Fully Connected Layers: The flattened features are then passed through a series of fully connected layers. These layers perform the final classification of the input image into one of the two categories (benign or malignant). The final layer uses a softmax activation function to produce a probability distribution over the two classes.

Output: The output of the model is the probability of the input image belonging to each of the two classes. The class with the highest probability is the predicted class for the input image.

To train the model, a dataset of brain scans with known labels (benign or malignant) is used. The VGG16 model is trained using backpropagation to adjust the weights of the model to minimize the error between the predicted and actual labels. Once the model is trained, it can be used to classify new brain scans as either benign or malignant.

