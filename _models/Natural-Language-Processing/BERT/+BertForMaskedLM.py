import torch
from transformers import BertTokenizer, BertForMaskedLM

text = "The [MASK] brown [MASK] jumps over the lazy dog."
# predict masked words
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertForMaskedLM.from_pretrained(bert_model_name)

# Tokenize input 
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
masked_indices = [i for i, token in enumerate(tokens) if token == '[MASK]']
token_ids = tokenizer.encode(text, return_tensors='pt')

# Get predicted logits for the masked tokens
with torch.no_grad():
    outputs = model(token_ids)
    predictions = outputs.logits

# Get the predicted tokens
predicted_tokens = [tokenizer.decode([torch.argmax(predictions[0, i]).item()]) for i in masked_indices]

predicted_text = text
for i, predicted_token in zip(masked_indices, predicted_tokens):
    predicted_text = predicted_text.replace('[MASK]', predicted_token, 1)  # Replace 1 occurrence at a time

print("Original Text:", text)
print("Predicted Tokens:", predicted_tokens)
print("Predicted Text:", predicted_text)
