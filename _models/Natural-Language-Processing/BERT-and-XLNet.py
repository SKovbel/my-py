from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel
import torch

# Load pre-trained BERT and XLNet models and tokenizers
bert_model_name = 'bert-base-uncased'
xlnet_model_name = 'xlnet-base-cased'

bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model_name)

bert_model = BertModel.from_pretrained(bert_model_name)
xlnet_model = XLNetModel.from_pretrained(xlnet_model_name)

# Example input text
text = "Once upon a time in a land far away."

# Tokenize input text for both BERT and XLNet
bert_inputs = bert_tokenizer(text, return_tensors='pt')
xlnet_inputs = xlnet_tokenizer(text, return_tensors='pt')

# Get embeddings from BERT
bert_outputs = bert_model(**bert_inputs)
bert_embeddings = bert_outputs.last_hidden_state

# Get embeddings from XLNet
xlnet_outputs = xlnet_model(**xlnet_inputs)
xlnet_embeddings = xlnet_outputs.last_hidden_state


xlnet_logits = bert_outputs.logits

# Concatenate the embeddings
concatenated_embeddings = torch.cat([bert_embeddings, xlnet_embeddings], dim=1)

# Now you can use the concatenated embeddings for downstream tasks
print(xlnet_embeddings)
