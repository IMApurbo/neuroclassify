```markdown
# NeuroClassify

NeuroClassify is a Python package for building and using image classification models using TensorFlow and Keras. It provides an easy-to-use interface for training models, saving and loading them, and making predictions on images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Creating an Image Classifier](#creating-an-image-classifier)
  - [Training the Model](#training-the-model)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
  - [Making Predictions](#making-predictions)
  - [Predicting Multiple Images](#predicting-multiple-images)
- [Functions](#functions)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install the NeuroClassify package using pip:

```bash
pip install neuroclassify
```

## Usage

### Creating an Image Classifier

To create an image classifier, you need to specify the base directory where your training images are stored. The images should be organized in subdirectories named after the class labels.

```python
from neuroclassify.classify import ImageClassifier

# Create an instance of the ImageClassifier
classifier = ImageClassifier(base_dir='path_to_your_images')
```

### Training the Model

To train the model, call the `train` method. You can specify the number of epochs for training.

```python
history = classifier.train(epochs=20)  # Train the model for 20 epochs
```

### Saving and Loading the Model

You can save the model using the `save_model` method, and load it later using the `load_model` method.

#### Save Model

```python
classifier.save_model(filename='my_model.h5')  # Save with a custom filename
```

#### Load Model

```python
classifier.load_model(filepath='my_model.h5')  # Load the model from the specified file
```

### Making Predictions

To predict the class of a single image, use the `predict_image` function. This function will return the predicted class name.

```python
from neuroclassify.utils import predict_image

# Get the class labels
class_labels = list(classifier.train_generator.class_indices.keys())

# Predict the class of an image
img_path = 'path_to_your_image/image.jpg'  # Path to the image
predicted_class_name = predict_image(classifier.model, img_path, class_labels)

print(f'Predicted Class: {predicted_class_name}')
```

### Predicting Multiple Images

You can also predict classes for all images in a directory using the `predict_images` function. This function will return a list of predicted class names.

```python
from neuroclassify.utils import predict_images

# Predict classes for all images in a directory
img_dir = 'path_to_your_image_directory'  # Directory containing images
predictions = predict_images(classifier.model, img_dir, class_labels)

for filename, predicted_class_name in predictions:
    print(f'File: {filename}, Predicted Class: {predicted_class_name}')
```

## Functions

### ImageClassifier Class
- **`__init__(self, base_dir, img_size=(150, 150), batch_size=32)`**: Initializes the image classifier with the base directory for images, image size, and batch size.
- **`train(self, epochs=20)`**: Trains the model for a specified number of epochs.
- **`save_model(self, filename='image_classifier_model.h5')`**: Saves the trained model to the specified filename.
- **`load_model(self, filepath)`**: Loads the model from the specified file.
- **`summary(self)`**: Prints the model summary.

### Utility Functions
- **`load_and_preprocess_image(img_path, target_size=(150, 150))`**: Loads and preprocesses an image for prediction.
- **`predict_image(model, img_path, class_labels, img_size=(150, 150))`**: Predicts the class of a single image.
- **`predict_images(model, img_dir, class_labels, img_size=(150, 150))`**: Predicts classes for all images in a specified directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or report issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Instructions to Customize
- Replace `path_to_your_images` and `path_to_your_image` with relevant paths or descriptions.
- You may want to add additional sections or modify existing ones to suit your project's needs.
- Include any specific instructions related to your project that may not be covered in this template.

