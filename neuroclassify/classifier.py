import os
import zipfile
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from .utils import save_labels

class ImageClassifier:
    def __init__(self, dataset_dir=None):
        self.dataset_dir = dataset_dir
        self.model = None
        self.class_indices = None
    
    def create_model(self, input_shape=(150, 150, 3), num_classes=2):
        """Creates a convolutional neural network model."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')  # Output layer
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model
    
    def train(self, epochs=10, batch_size=32):
        """Train the model with images from the dataset."""
        if not self.dataset_dir:
            raise ValueError("Dataset directory is not specified.")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'train'), target_size=(150, 150), batch_size=batch_size, class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'val'), target_size=(150, 150), batch_size=batch_size, class_mode='categorical'
        )
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stop]
        )
        
        self.class_indices = train_generator.class_indices
        save_labels(self.class_indices)

        # Plotting the training history
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
