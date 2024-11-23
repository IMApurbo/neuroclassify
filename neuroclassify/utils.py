def save_labels(class_indices, labels_file='labels.txt'):
    with open(labels_file, 'w') as f:
        for class_name, class_index in class_indices.items():
            f.write(f"{class_index}: {class_name}\n")
