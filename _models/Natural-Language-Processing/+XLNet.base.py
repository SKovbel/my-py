# Install the transformers library
from transformers import XLNetTokenizer, XLNetModel

# Load pre-trained XLNet model and tokenizer
xlnet_model_name = 'xlnet-base-cased'
xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model_name)
xlnet_model = XLNetModel.from_pretrained(xlnet_model_name, return_dict = True)

# Example input text
text = "Once upon a time in a land far away."

# Tokenize input text
input_ids = xlnet_tokenizer.encode(text, return_tensors='pt')

# Get embeddings from XLNet
outputs = xlnet_model(input_ids)
last_hidden_state = outputs.last_hidden_state
