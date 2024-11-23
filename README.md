# **NeuroClassify Usage Guide**

NeuroClassify is a Python package designed to help you easily create, train, save, and use image classification models with TensorFlow and Keras. The package provides functionality for training a model on a dataset, saving and loading the model, and making predictions on images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Creating an Image Classifier](#creating-an-image-classifier)
  - [Training the Model](#training-the-model)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
  - [Making Predictions](#making-predictions)
  - [Predicting Multiple Images](#predicting-multiple-images)
- [Functions](#functions)

---

## Installation

To install the `NeuroClassify` package, you can use pip:

```
pip install neuroclassify
```

---

## Usage

### **Creating an Image Classifier**

To create an image classifier, you need to specify the base directory where your training images are stored. The images should be organized into subdirectories named after the class labels. Here's how you can set up the classifier:

```python
from neuroclassify import ImageClassifier

# Create an instance of the ImageClassifier
classifier = ImageClassifier(img_size=(150, 150), batch_size=32)
classifier.dataset_dir = 'path_to_your_dataset'
```

In the above code, `path_to_your_dataset` should be the directory containing subdirectories for training and validation images.

### **Training the Model**

After creating the classifier, you can train it by calling the `train` method. You can specify the number of epochs for training:

```python
history = classifier.train(epochs=20)  # Train the model for 20 epochs
```

The method will automatically load the images from the `train` and `val` subdirectories inside your dataset directory and start the training process.

---

### **Saving and Loading the Model**

You can save the trained model to a file and load it back later using the following methods:

#### Save Model

To save the model, call the `save_model` method. This will save both the model in a `.h5` file and the class labels in a `labels.txt` file:

```python
classifier.save_model(name='my_model')  # Save with a custom filename
```

This will also create a `.zip` file that includes the model and the `labels.txt` file.

#### Load Model

To load the model, use the `load_model` method, providing the paths to the saved model and label files:

```python
classifier.load_model(model='my_model.h5', label='labels.txt')  # Load the model from the specified file
```

---

### **Making Predictions**

To make predictions with the trained model, you can use the `predict_image()` function. This allows you to classify a single image and get the predicted class name.

#### Predicting a Single Image

Here is how you can predict the class of a single image:

```python
# Define the path to your model, image, and labels
img_path = 'path_to_your_image/image.jpg'
classifier.load_model(model='my_model.h5', label='labels.txt')  # Load the model from the specified file

# Predict the class of an image and display the image
predicted_class_name = predict_image(img_path, display=True)
print(f'Predicted Class: {predicted_class_name}')
```

The `display=True` option will show the image along with the predicted class name using Matplotlib.

#### Predicting Multiple Images

To predict the classes of all images in a directory, you can use the `predict_images` function. This will return a list of predictions for each image:

```python
# Predict classes for all images in a directory
img_dir = 'path_to_your_image_directory'  # Directory containing images
predictions = predict_images(img_dir, display=True)

for filename, predicted_class_name in predictions:
    print(f'File: {filename}, Predicted Class: {predicted_class_name}')
```

This will go through all images in the specified directory and print the predictions for each one.

---

## Functions

### **`ImageClassifier` Class**
- **`create_model`**: Builds and compiles the image classification model.
- **`train`**: Trains the model using images from the dataset.
- **`save_model`**: Saves the trained model and its labels.
- **`load_model`**: Loads a pre-trained model and its labels.

### **`predict_image`**
Predicts the class of a single image and optionally displays it.
- **Arguments**:
  - `model`: The trained model to use for predictions.
  - `img_path`: Path to the image file to predict.
  - `label_file`: Path to the label file.
  - `display` (optional): If `True`, displays the image with its predicted class.

### **`predict_images`**
Predicts the classes for all images in a directory and optionally displays them.
- **Arguments**:
  - `model`: The trained model to use for predictions.
  - `img_dir`: Directory containing images to predict.
  - `label_file`: Path to the label file.
  - `display` (optional): If `True`, displays images with their predicted classes.

### **`load_labels`**
Loads class labels from a text file (used internally for predictions).

---

## Contributing

Contributions to the `NeuroClassify` package are welcome! If you have suggestions or improvements, feel free to submit a pull request.

---

## License

`NeuroClassify` is open source and available under the [MIT License](https://github.com/IMApurbo/neuroclassify/blob/main/LICENSE).

---

This guide provides everything you need to get started with `NeuroClassify` for image classification. You can now train models, save/load them, and use them for predictions with ease.

---
