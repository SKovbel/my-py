import torch
from transformers import BertTokenizer, BertForMultipleChoice

question = "What is the capital of France?"
choices = ["Paris", "Berlin", "London", "Madrid"]

# multiple-choice question answering
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertForMultipleChoice.from_pretrained(bert_model_name)

inputs = tokenizer(question, choices, return_tensors="pt", padding=True, truncation=True)

input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
labels = torch.tensor(1).unsqueeze(0)  # Batch size 1

outputs = model(input_ids, labels)

predicted_answer_index = torch.argmax(outputs.logits).item()
predicted_answer = choices[predicted_answer_index]

# Print the question, choices, and predicted answer
print("Question:", question)
print("Answer Choices:", choices)
print("Predicted Answer:", predicted_answer)
