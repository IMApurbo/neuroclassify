import os
import zipfile
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from .utils import save_labels, predict_images


class ImageClassifier:
    def __init__(self, dataset_dir=None):
        self.dataset_dir = dataset_dir
        self.model = None
        self.class_indices = None
    
    def create_model(self, input_shape=(150, 150, 3), num_classes=2):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(train_generator.num_classes, activation='softmax')  # Output layer
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model
    
    def train(self, epochs=10, batch_size=32):
        if not self.dataset_dir:
            raise ValueError("Dataset directory is not specified.")
        
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.dataset_dir + '/train', target_size=(150, 150), batch_size=batch_size, class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.dataset_dir + '/val', target_size=(150, 150), batch_size=batch_size, class_mode='categorical'
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stop]
        )
        self.class_indices = train_generator.class_indices
        save_labels(self.class_indices)
        return history
    
    def save_model(self, name='model'):
        if self.model is None:
            raise ValueError("Model is not created or trained.")
        
        model_path = f'{name}.h5'
        label_path = 'labels.txt'
        
        self.model.save(model_path)
        zip_filename = f'{name}.zip'
        
        # Create a zip file with the model and labels
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            zipf.write(model_path)
            zipf.write(label_path)
        
        # Remove the .h5 model and label file
        os.remove(model_path)
        os.remove(label_path)
    
    def load_model(self, model='model.h5', label='labels.txt'):
        self.model = load_model(model)
        with open(label, 'r') as f:
            self.class_indices = {line.strip().split(": ")[1]: line.strip().split(": ")[0] for line in f.readlines()}
        return self.model
