import torch
from transformers import BertForPreTraining, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForPreTraining.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Dummy pre-training data (replace with your actual data)
corpus = ["Bert is a powerful NLP model.", "Hugging Face is a great library for transformers."]
max_length = 512

# Tokenize and encode the training data
input_ids = []
attention_masks = []

for text in corpus:
    encoded_data = tokenizer.encode_plus(text, return_tensors='pt', max_length=max_length, truncation=True)
    
    input_id = encoded_data['input_ids']
    attention_mask = encoded_data['attention_mask']
    
    input_ids.append(input_id)
    attention_masks.append(attention_mask)

# Pad sequences to the maximum length
input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

# Create a DataLoader for training
train_dataset = TensorDataset(input_ids, attention_masks)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Pre-training loop
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        optimizer.zero_grad()
        input_ids, attention_masks = batch

        # Use BERT for pre-training
        outputs = model(input_ids, attention_mask=attention_masks, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')

# Save the pre-trained model
model.save_pretrained('./pretrained_bert')
