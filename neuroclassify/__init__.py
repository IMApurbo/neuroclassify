# neuroclassify/__init__.py

# Import the main components of the package for easier access

from .classifier import ImageClassifier
from .predict import predict_image, predict_images
from .utils import save_labels, load_labels


# __all__ to specify what gets imported when using 'from neuroclassify import *'
__all__ = ['ImageClassifier', 'predict_image', 'predict_images', 'save_labels', 'load_labels']
