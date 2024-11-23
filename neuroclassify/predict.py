from tensorflow.keras.preprocessing import image
import numpy as np
from .classifier import ImageClassifier

def predict_image(model, img_path, label_file):
    """Predict the class of a single image."""
    # Load image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image

    # Predict using the model
    predictions = model.predict(img_array)
    class_indices = load_labels(label_file)

    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = list(class_indices.keys())[predicted_class]

    return predicted_class_name

def predict_images(model, img_dir, label_file):
    """Predict the class of all images in a directory."""
    predictions = []
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        predicted_class_name = predict_image(model, img_path, label_file)
        predictions.append((filename, predicted_class_name))

    return predictions

def load_labels(label_file):
    """Load labels from a text file."""
    with open(label_file, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    return {i: label for i, label in enumerate(labels)}
