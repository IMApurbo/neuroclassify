import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

def predict_image(model_path, img_path, labels_path, display_option='name'):
    """
    Predict the class of a single image.

    Parameters:
    - model_path: str, path to the trained model (.h5 file)
    - img_path: str, path to the image to predict
    - labels_path: str, path to the label file containing class names
    - display_option: str, 'name' to show only the predicted class name, 'both' to show both image and predicted class name

    Returns:
    - predicted_class_name: str, the name of the predicted class
    """

    # Load the model
    model = load_model(model_path)
    
    # Load class labels
    with open(labels_path, 'r') as f:
        class_labels = {line.strip().split(": ")[1]: line.strip().split(": ")[0] for line in f.readlines()}
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)
    
    # Get the predicted class name
    predicted_class_name = class_labels.get(str(predicted_class_index[0]))

    # Display options
    if display_option == 'both':
        # Show both the image and predicted class name
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        plt.title(f'Predicted: {predicted_class_name}')
        plt.show()
    elif display_option == 'name':
        # Display only the predicted class name
        print(f'Predicted Class: {predicted_class_name}')
    
    return predicted_class_name


def predict_images(model_path, folder_path, labels_path, display_option='name'):
    """
    Predict the classes of all images in a directory.

    Parameters:
    - model_path: str, path to the trained model (.h5 file)
    - folder_path: str, directory containing images to predict
    - labels_path: str, path to the label file containing class names
    - display_option: str, 'name' to show only the predicted class name, 'both' to show both image and predicted class name

    Returns:
    - predictions: dict, keys are image filenames, values are predicted class names
    """

    predictions = {}

    # Iterate through each image in the folder
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.endswith(('.jpg', '.png', '.jpeg')):  # Check for image file extensions
            predicted_class_name = predict_image(model_path, img_path, labels_path, display_option)
            predictions[img_name] = predicted_class_name
    
    return predictions
