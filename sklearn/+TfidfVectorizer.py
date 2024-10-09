from sklearn.feature_extraction.text import TfidfVectorizer
sentences = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    'Is this the first second?',
]
vectorizer = TfidfVectorizer(max_features=20)
X = vectorizer.fit_transform(sentences).toarray()
Y = vectorizer.get_feature_names_out()
print(X)
print(Y)