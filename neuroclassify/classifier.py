import os
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from .utils import save_labels, load_labels  # Ensure utils has save_labels and load_labels functions

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
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

        # Train the model
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.val_generator.batch_size,
            epochs=epochs,
            callbacks=[reduce_lr, early_stop]
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
        """Save the trained model and labels."""
        model_file = name + '.h5'
        self.model.save(model_file)
        with zipfile.ZipFile(f'{name}.zip', 'w') as zipf:
            zipf.write(model_file)
            zipf.write('labels.txt')
        os.remove(model_file)

    def load_model(self, model_file, label_file):
        """Load the model and labels."""
        self.model = load_model(model_file)
        self.class_indices = load_labels(label_file)
