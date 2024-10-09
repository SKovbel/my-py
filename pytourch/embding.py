import torch
import torch.nn as nn

sentence = "Word1 word2 word3 word4".lower()
vocab = {"word1": 0, "word2": 1, "word3": 2, "word4": 3}

sentence_tokens = [vocab[word] for word in sentence.split()] 
sentence_tensor = torch.tensor(sentence_tokens).unsqueeze(0) 

vocab_size = len(vocab)
embedding_dim = 5

embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
embedded_sentence = embedding_layer(sentence_tensor)

print(f"Tokenized sentence: {sentence_tokens}")
print(f"Shape of embedded sentence: {embedded_sentence.shape}") 
print(f"Embedded sentence:\n{embedded_sentence}")
