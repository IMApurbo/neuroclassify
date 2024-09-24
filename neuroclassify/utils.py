import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

def predict_image(model, img_path, class_labels, img_size=(150, 150)):
    img_array = load_and_preprocess_image(img_path, target_size=img_size)
    predictions = model.predict(img_array)  # Get predictions
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_labels[predicted_class]

    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class_name}')
    plt.axis('off')
    plt.show()

    return predicted_class_name
