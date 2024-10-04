from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

lm = 'google/flan-t5-base'
lang_model = T5ForConditionalGeneration.from_pretrained(lm)
lang_model.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False)

questions = open('qa/questions.txt').readlines()
answers = open('qa/answers.txt').readlines()
#questions = [
#"What is the most populated country in the world?",
#"What is a boy to his mom?",
#"Which country lost second world war?",
#"What city is called 'The big apple'?",
#"What country was Chistopher Columbus looking for when he discovered America?",
#]
#answers = [
#"India.",
#"Her son.",
#"Germany.",
#"New York.",
#"India.",
#]

prefixes = ['The following question is about Star Wars:']
postfixes = ['The answer to the question is:']
#prefixes = ['']
#postfixes = ['']

def eval(gold, pred):
    """
    An answer is considered correct if at least half of the gold
    words are in the prediction.
    """
    gold = set(gold.strip().lower().replace('.', '').split(' '))
    pred = set(pred.strip().lower().replace('.', '').split(' '))
    return len(gold.intersection(pred)) >= len(gold)/2


for prefix, postfix in zip(prefixes, postfixes):
    correct = 0
    for question, answer in zip(questions, answers):
        question = prefix + ' ' + question.strip() + ' ' + postfix
        tokked = tokenizer(question.strip(), return_tensors='pt')['input_ids']
        tokked = tokked.to(DEVICE)
        generated_ids = lang_model.generate(tokked, max_new_tokens=20)
        tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(question)
        print(' '.join(tokens))
        print()
        correct += int(eval(answer, ' '.join(tokens)))

    print(str(correct) + ' out of ' + str(len(answers)) + ' correct')


