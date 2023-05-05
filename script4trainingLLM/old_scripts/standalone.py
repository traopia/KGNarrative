from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,Seq2SeqTrainer
import os
from datasets import load_dataset
import numpy as np
from utils import *
import torch
import evaluate
import sys
import json
import time
import argparse

def tokenize_for_evaluation(tokenizer,preds,labels):


    predicted_text = []
    golden_labels = []

    for pred, label in zip(preds, labels):

        gen = tokenizer.decode(pred, skip_special_tokens=True)
        gen = str(gen)
        predicted_text.append(gen)

        gold = tokenizer.decode(label, skip_special_tokens=True)
        gold = str(gold)
        golden_labels.append(gold)

        return predicted_text,golden_labels
    
def process_data_BART(data_to_process,tokenizer,max_input,max_target,typeKG ):

    #get the dialogue text
    inputs = [graph for graph in data_to_process[f'{typeKG}']]
    #tokenize text
    model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)

    #tokenize labels
    #with tokenizer.as_target_tokenizer():
    targets = [target for target in data_to_process['story']]
    model_targets = tokenizer(targets, max_length=max_target, padding='max_length', truncation=True)
    

    #reuturns input_ids, attention_masks, labels
    
    data_to_process["input_ids"] = model_inputs.input_ids
    data_to_process["attention_mask"] = model_inputs.attention_mask
    data_to_process["labels"] = model_targets.input_ids

    return data_to_process

    
datapath ='/daatapath
dataprefix ='pop'
typeKG = 'Instances_KG'
model_checkpoint="facebook/bart-base"
experiment_name = 'exp'
learning_rate =1e-4
batch_size = 1
epochs =3
save_model = False
max_target = 512
max_input = 512


train_file = datapath +'/' + dataprefix + '_train' + '.json'
dev_file = datapath +'/'+ dataprefix + '_dev' + '.json'
test_file = datapath +'/' + dataprefix + '_test'+ '.json'


print("Loading dataset from ",datapath)
dataset = load_dataset('json', data_files={'train': train_file, 'valid': dev_file, 'test': test_file})

todrop=list(set(dataset['test'].column_names)-set([typeKG,'story'])) #This line returns a list of all the columns to drop (all columns minus the ones we need (input typeKG and story))


print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_eos_token=True)

print("\nProcessing Dataset")
#the processing of the data is done batches for make it faster,number of processes 4
tokenized_dataset = dataset.map(lambda example: process_data_BART(example, tokenizer,max_input,max_target,typeKG), batched=True, num_proc=4,remove_columns=todrop)

print("\nLoading MODEL")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
#model.to(device)

print("Collator for batches")
collator = DataCollatorForSeq2Seq(tokenizer, model=model) #this is necessary for diving in batch for training

print('Loading rouge')
rouge = evaluate.load('rouge')


def compute_rouge(pred): 
    predictions, labels = pred
    #decode the predictions
    decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #decode labels
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True,clean_up_tokenization_spaces=True)

    #compute results
    res = rouge.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
    #get %
    return res

print("\nPREPARING FOR TRAINING...")

#defining training arogouments
args = Seq2SeqTrainingArguments(
    experiment_name,
    evaluation_strategy='epoch',
    learning_rate=learning_rate, 
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size= batch_size,
    gradient_accumulation_steps=3, #compute gradient on n examples KG story 
    weight_decay=0.01, #regularization
    save_total_limit=1, #this is the max amount of checkpoint saved, after which previous checpoints are removed
    num_train_epochs=epochs, #number of epochs
    predict_with_generate=True, 
    generation_max_length = 512, #max number of tokens per generation 
    generation_num_beams=5, #decoding strategy! greedy search, beam search 
    eval_accumulation_steps=1, #backprop  
    fp16=True, #memory management
    disable_tqdm=True)
#only CUDA available -> fp16=True


### almost training time
trainer = Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['valid'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge
)


trainer.train()


if save_model:
    print("Saving model")
    trainer.save_model(experiment_name+"/saved_model")


print("\nPREDICTING..")
preds, labels, metrics = trainer.predict(tokenized_dataset['test'], num_beams=5, min_length=50, max_length=512, no_repeat_ngram_size=2, early_stopping=True)

predicted_text,golden_labels=tokenize_for_evaluation(tokenizer,preds,labels)

#here is already past the error 
print("\nRESULT SCORES:")

scores = metrics.items()
print(f'Results: {scores}')
```