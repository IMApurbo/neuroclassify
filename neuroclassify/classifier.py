import os
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from .utils import save_labels, load_labels  # Ensure utils has save_labels and load_labels functions
import json  # To save and load labels as JSON
import numpy as np
from tensorflow.keras.preprocessing import image


class ImageClassifier:
    def __init__(self, img_size=(150, 150), batch_size=32):
        """Initialize ImageClassifier without dataset directory."""
        self.dataset_dir = None  # Dataset directory set later
        self.img_size = img_size
        self.batch_size = batch_size
        self.datagen = None
        self.train_generator = None
        self.val_generator = None
        self.model = None
        self.class_indices = None

    def setup_generators(self):
        """Set up data generators using the provided dataset directory."""
        if not self.dataset_dir or not os.path.exists(self.dataset_dir):
            raise ValueError("Dataset directory is not set or does not exist!")

        # Data augmentation and splitting
        self.datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=0.2  # 20% for validation
        )

        # Training generator
        self.train_generator = self.datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )

        # Validation generator
        self.val_generator = self.datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            seed=42
        )

    def build_model(self):
        """Define the CNN model architecture."""
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_generator.class_indices), activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train(self, epochs=20):
        """Train the CNN model."""
        if not self.train_generator or not self.val_generator:
            self.setup_generators()  # Ensure generators are set up

        if not self.model:
            self.build_model()  # Ensure model is built

        # Learning rate scheduler and early stopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        # Train the model
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.val_generator.batch_size,
            epochs=epochs,
            callbacks=[reduce_lr]
        )

        # Save class indices
        self.class_indices = self.train_generator.class_indices
        save_labels(self.class_indices)

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """Plot the training and validation accuracy/loss."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.show()

    def save_model(self, name='model'):
        """
        Save the trained model and labels to a zip file.
        Args:
            name (str): The base name for the saved files (without extension).
        """
        # Save the model to an H5 file
        model_file = name + '.h5'
        self.model.save(model_file)
    
        # Save the labels to a JSON file using the updated save_labels function
        labels_file = 'labels.json'
        save_labels(self.class_indices, labels_file)

        # Create a zip file containing the model and labels
        with zipfile.ZipFile(f'{name}.zip', 'w') as zipf:
            zipf.write(model_file)
            zipf.write(labels_file)
    
        # Clean up by removing the model and labels files
        os.remove(model_file)
        os.remove(labels_file)
        print(f"Model and labels saved to {name}.zip")

    def load_model(self, model_file, label_file):
        """Load the trained model and the class labels."""
        self.model = load_model(model_file)
        self.class_indices = self.load_labels(label_file)

    def load_labels(self, label_file):
        """Load the class labels from a JSON file."""
        with open(label_file, 'r') as f:
            labels_dict = json.load(f)
        return labels_dict

def predict_image(self, img_path, display=False):
    """
    Predict the class label for a single image and optionally display it.
    Args:
        img_path (str): Path to the image file.
        display (bool): Whether to display the image with the prediction.
    Returns:
        str: Predicted class label.
    """
    if not self.class_indices:
        raise ValueError("Class indices are not loaded. Load the labels first.")

    # Invert the class_indices dictionary to map indices back to labels
    index_to_label = {v: k for k, v in self.class_indices.items()}

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=self.img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict the class
    predictions = self.model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])

    # Map the predicted class index to the class label
    predicted_class_label = index_to_label.get(predicted_class_idx, "Unknown class")

    # Optionally display the image with the predicted label
    if display:
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class_label}")
        plt.axis('off')
        plt.show()

    return predicted_class_label

def predict_images(self, folder_path, display=False):
    """
    Predict class labels for all images in a folder and optionally display them.
    Args:
        folder_path (str): Path to the folder containing image files.
        display (bool): Whether to display the images with predictions.
    Returns:
        dict: A dictionary with image paths as keys and predicted class labels as values.
    """
    if not self.class_indices:
        raise ValueError("Class indices are not loaded. Load the labels first.")

    # Invert the class_indices dictionary to map indices back to labels
    index_to_label = {v: k for k, v in self.class_indices.items()}

    predictions = {}

    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Loop through each image file in the folder
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Predict the class
        predictions_array = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions_array[0])

        # Map the predicted class index to the class label
        predicted_class_label = index_to_label.get(predicted_class_idx, "Unknown class")

        # Store the prediction
        predictions[img_path] = predicted_class_label

        # Optionally display the image with the predicted label
        if display:
            plt.imshow(image.load_img(img_path))
            plt.title(f"Predicted: {predicted_class_label}")
            plt.axis('off')
            plt.show()

    return predictions

