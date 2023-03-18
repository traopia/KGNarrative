from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,Seq2SeqTrainer
import os
from datasets import load_dataset, load_metric
import numpy as np
from utils import *
import torch
import evaluate
import sys
import json
import time
os.environ['TQDM_DISABLE'] = 'true'

def main(argv, arc):

    if arc!=6:
        print(" ARGUMENT USAGE IS WRONG, RUN FILE LIKE: finetune_bart.py [datapath] [dataset] [type] [model folder] [Experiment_name]")
        exit()

    dataset = argv[2]
    datapath = argv[1] 
    typeKG = argv[3]
    model_name=argv[4]
    experiment_name=argv[5]
    tokenizer_name='facebook/bart-base'

    #CUDA CHECK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device in use is ", device)


    train_file = datapath +'/'+ dataset +'_train.json'
    dev_file = datapath +'/'+  dataset + '_validation.json'
    test_file = datapath +'/'  + dataset + '_test.json'

    print("Loading dataset from",train_file)
    dataset = load_dataset('json', data_files={'train': train_file, 'valid': dev_file, 'test': test_file})


    print("\nLoading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("\nProcessing Dataset")
    #the processing of the data is done batches for make it faster,number of processes 4
    tokenized_dataset = dataset.map(lambda example: process_data_BART(example, tokenizer,max_input,max_target,typeKG), batched=True, num_proc=4,remove_columns=['Instance_Knowledge_Graph', 'story'])

    print("\nLoading MODEL")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    print("Collator for batches")
    collator = DataCollatorForSeq2Seq(tokenizer, model=model) #this is necessary for diving in batch for training

    print('Loading rouge')
    metric = load_metric('rouge')

    def compute_rouge(pred): #UGLY AND DEPPRECATED
        predictions, labels = pred
        #decode the predictions
        decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #decode labels
        decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        #compute results
        res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
        #get %
        res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

        pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        res['gen_len'] = np.mean(pred_lens)

        return {k: round(v, 4) for k, v in res.items()}

    print("\nPREPARING FOR TRAINING...")




if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
