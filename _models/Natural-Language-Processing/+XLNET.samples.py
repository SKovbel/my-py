# https://towardsdatascience.com/how-to-use-xlnet-from-the-hugging-face-transformer-library-ddd0b7c8d0b9
import torch
import torch.nn as nn 
from transformers import XLNetModel, XLNetTokenizer, XLNetLMHeadModel, XLNetForMultipleChoice, XLNetForQuestionAnsweringSimple 
from torch.nn import functional as F


def case1():
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

def case2():
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

def case3(text, deep = 3):
    # predicting the best word to follow/continue a sentence
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased', return_dict = True)
    text = text + tokenizer.mask_token + " "
    input = tokenizer.encode_plus(text, return_tensors = "pt")
    logits = model(**input).logits
    output = logits[:, -1, :]
    softmax = F.softmax(output, dim = -1)
    index = torch.argmax(softmax, dim = -1)
    x = tokenizer.decode(index)
    new_sentence = text.replace(tokenizer.mask_token, x)
    if deep > 0:
        case3(new_sentence, deep - 1)
    else:
        print(new_sentence)


def case4():
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


class XLNet_Model(nn.Module): 
    def __init__(self, classes): 
        super(XLNet_Model, self).__init__() 
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased) ') 
        self.out = nn.Linear(self.xlnet.config.hidden_size, classes) 
    def forward(self, input): 
        outputs = self.xlnet(**input) 
        out = self.out(outputs.last_hidden_state) 
        return out


def case5():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNet_Model(XLNetLMHeadModel)
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

case5()
case1()
case2()
case3("The sky is very clear when ", 5)
case4()


