import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load pre-trained XLNet model and tokenizer
model_name = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetLMHeadModel.from_pretrained(model_name)

# Custom dataset class for pretraining
class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Path to your training data file
train_data_path = '_models/Natural-Language-Processing/training_data.txt'

# Create a DataLoader for training
train_dataset = CustomDataset(train_data_path)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        optimizer.zero_grad()

        # Tokenize and encode the batch
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True)

        # Forward pass
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')

# Save the pretrained model
model.save_pretrained('tmp/pretrained_xlnet')


def case3(text):
    # predicting the best word to follow/continue a sentence
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased', return_dict = True)
    text = text + tokenizer.mask_token
    input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
    output = model(**input).logits
    softmax = F.softmax(output, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

case3("This is a simple sentence for ")
