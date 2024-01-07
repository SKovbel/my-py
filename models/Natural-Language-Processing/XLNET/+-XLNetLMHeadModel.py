# https://towardsdatascience.com/how-to-use-xlnet-from-the-hugging-face-transformer-library-ddd0b7c8d0b9
import torch
import torch.nn as nn 
from transformers import XLNetTokenizer, XLNetLMHeadModel
from torch.nn import functional as F

def case3(text, deep = 3):
    # predicting the best word to follow/continue a sentence
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased', return_dict = True)
    text = "The sky is very clear at " + tokenizer.mask_token
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

case3("The sky is very clear when ", 5)


