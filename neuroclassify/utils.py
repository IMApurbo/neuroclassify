import os
import json
import zipfile

def save_labels(class_indices, labels_file='labels.json'):
    """
    Save the class labels to a JSON file in the format {index: label}.
    Args:
        class_indices (dict): A dictionary where keys are class names and values are their indices.
        labels_file (str): The name of the output JSON file.
    """
    labels_dict = {str(index): label for label, index in class_indices.items()}
    with open(labels_file, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    print(f"Labels saved to {labels_file} in the required format.")

def load_labels(label_file):
    """
    Load labels from a JSON file.
    Args:
        label_file (str): The JSON file containing the labels.
    Returns:
        dict: A dictionary with labels loaded from the file.
    """
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file {label_file} not found.")
    with open(label_file, 'r') as f:
        labels_dict = json.load(f)
    return labels_dict

def save_model_to_zip(model, filename):
    """
    Save the model and labels to a zip file.
    Args:
        model: The trained model to save.
        filename (str): The base name of the output files (without extensions).
    """
    # Save the model to an H5 file
    model_file = f"{filename}.h5"
    model.save(model_file)
    
    # Create a zip file containing the model and labels
    with zipfile.ZipFile(f'{filename}.zip', 'w') as zipf:
        zipf.write(model_file)
        zipf.write('labels.json')
    
    # Optional: Clean up by removing the standalone model file
    os.remove(model_file)
    print(f"Model and labels saved to {filename}.zip")
