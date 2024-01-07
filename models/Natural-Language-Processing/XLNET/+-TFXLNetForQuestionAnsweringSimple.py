# https://towardsdatascience.com/how-to-use-xlnet-from-the-hugging-face-transformer-library-ddd0b7c8d0b9
import torch
import torch.nn as nn 
from transformers import XLNetTokenizer, XLNetForQuestionAnsweringSimple 
from torch.nn import functional as F

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# answering a question given some context text by outputting the start and end indexes
model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased", return_dict = True)

question = "How many continents are there in the world?"
text = "There are 7 continents in the world."

inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
output = model(**inputs)

start_max = torch.argmax(F.softmax(output.start_logits, dim = -1))
end_max = torch.argmax(F.softmax(output.end_logits, dim=-1)) + 1 

answer = tokenizer.decode(inputs["input_ids"][0][start_max : end_max])
print(answer)
