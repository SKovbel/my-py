import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding

# Example sentences
sentences = [
    #"400.0HP 5.6L 8 Cylinder Engine Gasoline Fuel",
    "4.0L V8 32V GDI DOHC Twin Turbo",
    "4.0L V8 32V GDI DOHC Turbo Twin"
]

max_tokens = 10000
embedding_dim = 6
max_length = 10

vectorize_layer = TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=max_length)
vectorize_layer = TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=None)
vectorize_layer = TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=max_length)

embedding_layer = Embedding(input_dim=max_tokens, output_dim=embedding_dim, input_length=max_length)

vectorize_layer.adapt(sentences)

vectorized_sentences = vectorize_layer(sentences)

embedded_sentences = embedding_layer(vectorized_sentences)

print(f"Vectorized sentence 1 shape: {embedded_sentences[0].shape}")
print(f"Vectorized sentence 2 shape: {embedded_sentences[1].shape}")

print("Embedding for original sentence:")
print(embedded_sentences[0]) 

print("\nEmbedding for modified sentence:")
print(embedded_sentences[1]) 
