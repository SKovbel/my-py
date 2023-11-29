import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Example input text
text = "Once upon a time in a land far away."

# Tokenize input text
input_ids = bert_tokenizer.encode(text, return_tensors='pt')

# Get embeddings from BERT
outputs = bert_model(input_ids)
embeddings = outputs.last_hidden_state

text_from_embeddings = bert_tokenizer.decode(outputs, skip_special_tokens=True)

# Print the original text and the text from embeddings
print("Original Text:", text)
print("Text from Embeddings:", text_from_embeddings)