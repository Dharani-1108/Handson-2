!pip install transformers datasets pillow
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests

# Load model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
