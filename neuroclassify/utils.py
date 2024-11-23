import os
import json
import zipfile

def save_labels(class_indices, labels_file='labels.json'):
    """Save the class labels to a JSON file in the format {index: label}."""
    labels_dict = {str(i): label for i, label in enumerate(class_indices.values())}
    with open(labels_file, 'w') as f:
        json.dump(labels_dict, f, indent=2)

def load_labels(label_file):
    """Load labels from a JSON file."""
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file {label_file} not found.")
    with open(label_file, 'r') as f:
        labels_dict = json.load(f)
    return labels_dict

def save_model_to_zip(model, filename):
    """Save the model to a zip file."""
    model.save(filename)
    with zipfile.ZipFile(f'{filename}.zip', 'w') as zipf:
        zipf.write(filename)
        zipf.write('labels.json')
