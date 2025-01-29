!pip install transformers datasets pillow

#Step 4: Load Pre-Trained Model for Image Classification
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests

# Load model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests

# Load model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

#Step 5: Load and Process an Image
# Load an image from the web
url = "/content/image.jpg"  # Replace with an actual image URL
# Instead of using requests.get for a local file, use Image.open directly
image = Image.open("/content/image.jpg") # Changed to direct file path
image = image.convert("RGB")

# Preprocess image
inputs = feature_extractor(images=image, return_tensors="pt")


#Step 6: Perform Inference
# Perform inference
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
# Map predicted class index to class label
predicted_class = model.config.id2label[predicted_class_idx]
print("Predicted class:", predicted_class)

#Step 7: Explain Image Classification Using LLM
from transformers import pipeline

# Load a text-generation model (GPT-like)
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Create a prompt
explanation_prompt = f"The Vision Transformer (ViT) model classified the given image as class '{predicted_class}'. "
explanation_prompt += "Explain why this classification is reasonable based on the image's features."

from transformers import pipeline

# Load a text-generation model (GPT-like)
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Create a prompt
explanation_prompt = f"The Vision Transformer (ViT) model classified the given image as class '{predicted_class}'. "
explanation_prompt += "Explain why this classification is reasonable based on the image's features."

# Generate explanation
llm_explanation = llm(explanation_prompt, max_length=150)[0]["generated_text"]
from transformers import pipeline

# Load a text-generation model (GPT-like)
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Create a prompt
explanation_prompt = f"The Vision Transformer (ViT) model classified the given image as class '{predicted_class}'. "
explanation_prompt += "Explain why this classification is reasonable based on the image's features."

# Generate explanation
llm_explanation = llm(explanation_prompt, max_length=150)[0]["generated_text"]

print("\n🔍 LLM Explanation:\n", llm_explanation)
