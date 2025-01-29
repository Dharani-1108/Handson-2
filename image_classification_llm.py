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
