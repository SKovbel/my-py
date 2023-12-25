import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def get_top_answers(possible_starts,possible_ends,input_ids):
  answers = []
  for start,end in zip(possible_starts,possible_ends):
        #+1 for end
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))
        answers.append( answer )
  return answers  

def answer_question(question,context,topN):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")    
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    model_out = model(**inputs)
     
    answer_start_scores = model_out["start_logits"]
    answer_end_scores = model_out["end_logits"]

    possible_starts = np.argsort(answer_start_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
    possible_ends = np.argsort(answer_end_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
    
    #get best answer
    answer_start = torch.argmax(answer_start_scores)  
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    answers = get_top_answers(possible_starts,possible_ends,input_ids )

    return { "answer":answer,"answer_start":answer_start,"answer_end":answer_end,"input_ids":input_ids,
            "answer_start_scores":answer_start_scores,"answer_end_scores":answer_end_scores,"inputs":inputs,"answers":answers,
            "possible_starts":possible_starts,"possible_ends":possible_ends}




answer_question("The capital of France is Paris. It is known for its art, culture, and history.", "What is the capital of France?", 2)
