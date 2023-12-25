# https://towardsdatascience.com/how-to-use-xlnet-from-the-hugging-face-transformer-library-ddd0b7c8d0b9
import torch
import torch.nn as nn 
from transformers import XLNetTokenizer, XLNetForMultipleChoice
from torch.nn import functional as F


tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# multiple-choice question answering
model = XLNetForMultipleChoice.from_pretrained("xlnet-base-cased", return_dict = True)

prompt = "What is the capital of France?"
answers = ["Paris", "London", "Lyon", "Berlin"]

encoding = tokenizer([prompt, prompt, prompt, prompt], answers, return_tensors="pt", padding = True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}) 

logits = outputs.logits
softmax = F.softmax(logits, dim = -1)
index = torch.argmax(softmax, dim = -1)
print("The correct answer is", answers[index])
