import torch
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2ForSequenceClassification.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Dummy data (replace with your actual data)
context = "GPT-2 is a large language model."
question = "What is GPT-2?"

# Combine context and question into a single string
input_text = f"Context: {context} Question: {question}"

# Tokenize and encode the input
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Forward pass to get logits
logits = model(input_ids).logits

# Get the predicted class (answer)
predicted_class = torch.argmax(logits, dim=1).item()

# Print the result
print("Question:", question)
print("Answer:", predicted_class)
