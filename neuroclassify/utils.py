import os

def save_labels(class_indices, labels_file='labels.txt'):
    """Save the class labels to a text file."""
    labels = [class_indices[key] for key in sorted(class_indices.keys())]
    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(f'{label}\n')

def load_labels(label_file):
    """Load labels from a text file."""
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file {label_file} not found.")
    with open(label_file, 'r') as f:
        labels = f.readlines()
    return {i: label.strip() for i, label in enumerate(labels)}

def save_model_to_zip(model, filename):
    """Save the model to a zip file."""
    model.save(filename)
    with zipfile.ZipFile(f'{filename}.zip', 'w') as zipf:
        zipf.write(filename)
        zipf.write('labels.txt')
