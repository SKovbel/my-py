import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Example sentence
sentence = "SpaCy is a powerful natural language processing library. Hellow how are you?"

# Process the sentence using spaCy
doc = nlp(sentence)

# Print the part-of-speech tags and dependency relationships
for sentence in doc.sents:
    print(sentence.text)
    for token in sentence:
        print(f"{token.pos_}", end=" ")
    print()


import pandas as pd

# Example DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'Los Angeles']}

df = pd.DataFrame(data)

# Iterate over all rows
for index, row in df.iterrows():
    print(f"Index: {index}, Name: {row['Name']}, Age: {row['Age']}, City: {row['City']}")