```markdown
# NeuroClassify

NeuroClassify is a simple and user-friendly image classification package built with TensorFlow. It allows users to train deep learning models on their own image datasets and make predictions on new images with ease.

## Features

- Train a custom image classification model using TensorFlow.
- Predict classes for single images or multiple images from a directory.
- Support for data augmentation to enhance model performance.
- Save and load trained models easily.

## Installation

You can install the NeuroClassify package using pip. Simply run:

```
pip install neuroclassify
```

## Usage

Here’s a quick guide on how to use NeuroClassify for image classification tasks.

### Importing the Package

```python
from neuroclassify import ImageClassifier
```

### Initializing the Classifier

Create an instance of the `ImageClassifier` class by specifying the base directory containing your dataset:

```python
classifier = ImageClassifier(base_dir='path/to/your/dataset')
```

### Training the Model

You can train the model by calling the `train` method. You can specify the number of epochs for training. The default value is set to 20 epochs.

```python
classifier.train(epochs=30)  # To train the model for 30 epochs
```

### Making Predictions

#### Predicting a Single Image

To predict the class of a single image, use the `predict_image` method:

```python
predicted_class = classifier.predict_image(img_path='path/to/image.jpg')
print(f'Predicted class: {predicted_class}')
```

#### Predicting Multiple Images

To predict the classes of all images in a directory, use the `predict_images` method:

```python
predicted_classes = classifier.predict_images(img_dir='path/to/image/directory')
print(predicted_classes)  # List of predicted classes for each image
```

### Saving and Loading Models

You can save your trained model using the `save_model` method:

```python
classifier.save_model(filepath='path/to/save/model.h5')
```

To load a previously saved model, use the `load_model` method:

```python
classifier.load_model(filepath='path/to/saved/model.h5')
```

## Examples

Here are some examples of how you can use the NeuroClassify package.

### Example 1: Basic Usage

```python
from neuroclassify.classifier import ImageClassifier

# Initialize the classifier
classifier = ImageClassifier(base_dir='path/to/your/dataset')

# Train the model
classifier.train(epochs=30)

# Predict a single image
predicted_class = classifier.predict_image(img_path='path/to/image.jpg')
print(f'Predicted class: {predicted_class}')

# Predict multiple images in a directory
predicted_classes = classifier.predict_images(img_dir='path/to/image/directory')
print(predicted_classes)
```

### Example 2: Saving and Loading Models

```python
from neuroclassify.classifier import ImageClassifier

# Initialize the classifier
classifier = ImageClassifier(base_dir='path/to/your/dataset')

# Train the model
classifier.train(epochs=30)

# Save the trained model
classifier.save_model(filepath='my_model.h5')

# Load the saved model
classifier.load_model(filepath='my_model.h5')

# Predict an image using the loaded model
predicted_class = classifier.predict_image(img_path='path/to/image.jpg')
print(f'Predicted class after loading: {predicted_class}')
```

## Contributing

Contributions are welcome! If you'd like to contribute to NeuroClassify, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out to:

- **Your Name** - [bandinvisible8@gmail.com](mailto:bandinvisible8@gmail.com)

Thank you for using NeuroClassify!
```
