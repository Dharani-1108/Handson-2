!pip install transformers datasets pillow

# Step 4: Load Pre-Trained Model for Image Classification
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
import torchvision.transforms as transforms

# Load model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Step 5: Feature Engineering - Image Preprocessing
# Load an image from the web
url = "/content/image.jpg"  # Replace with an actual image URL
image = Image.open("/content/image.jpg")  # Changed to direct file path
image = image.convert("RGB")

# Apply basic transformations: resize and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Apply the transformations to the image
image_transformed = transform(image)

# Step 6: Preprocess Image for Model Inference
inputs = feature_extractor(images=image_transformed, return_tensors="pt")

# Step 7: Perform Inference
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
# Map predicted class index to class label
predicted_class = model.config.id2label[predicted_class_idx]
print("Predicted class:", predicted_class)

# Step 8: Explain Image Classification Using LLM
from transformers import pipeline

# Load a text-generation model (GPT-like)
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Create a prompt
explanation_prompt = f"The Vision Transformer (ViT) model classified the given image as class '{predicted_class}'. "
explanation_prompt += "Explain why this classification is reasonable based on the image's features."

# Generate explanation
llm_explanation = llm(explanation_prompt, max_length=150)[0]["generated_text"]

print("\n🔍 LLM Explanation:\n", llm_explanation)
