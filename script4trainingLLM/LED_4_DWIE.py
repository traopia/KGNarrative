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
os.environ['TQDM_DISABLE'] = 'true'

max_target = 1024
max_input = 4096


def add_args(parser):
    parser.add_argument('datapath', type=str, help='Path to the data directory')
    parser.add_argument('dataset', type=str, help='prefix of the dataset')
    parser.add_argument('graph_kind', type=str, help='Kind of graph')
    parser.add_argument('model_checkpoint', type=str, help='HF MODELS OR Path to the directory containing the model checkpoint files')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment (outputfolder)')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the optimizer (default: 2e-4)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--save_model', type=bool, default=False, help='Save the model (default: False)')
    return parser

def eval_pipeline(predicted_text,golden_labels,metrics,graph_for_parent):

    print("\nRESULT SCORES:")
    score_to_print=[]

    scores = metrics.items()
    print(f'Results: {scores}')
    score_to_print.append(metrics)

    bleu = evaluate.load("bleu")
    result_bleu= bleu.compute(predictions=predicted_text, references=golden_labels)
    print(f'{result_bleu=}')
    score_to_print.append(result_bleu)

    google_bleu = evaluate.load("google_bleu")
    result_google_bleu = google_bleu.compute(predictions=predicted_text, references=golden_labels)
    print(f'{result_google_bleu=}')
    score_to_print.append(result_google_bleu)

    meteor = evaluate.load("meteor")
    result_meteor= meteor.compute(predictions=predicted_text, references=golden_labels)
    print(f'{result_meteor=}')
    score_to_print.append(result_meteor)

    bertscore = evaluate.load("bertscore")
    results_bert = bertscore.compute(predictions=predicted_text, references=golden_labels, model_type="distilbert-base-uncased")
    results_bert={"Bert_Score":{i:np.mean(results_bert[i]) for i in list(results_bert.keys())[:-1]}}#this line is bc there is an hashvalue in results_bert that we dont need thus we only take first 3 elemnts of dictionary and avg 
    print(f'{results_bert=}')
    score_to_print.append(results_bert)

    bleurt = evaluate.load("bleurt",'BLEURT-20',module_type="metric")
    result_bleurt = bleurt.compute(predictions=predicted_text, references=golden_labels)
    result_bleurt["bleurt_score"] = np.mean(result_bleurt.pop("scores"))
    print(f"{result_bleurt=}")
    score_to_print.append(result_bleurt)

        
    graph_for_parent=[g.split('[TRIPLES]')[1] for g in graph_for_parent] #this because the isntance graph has a the core
    #print("len of graph for parent", len(graph_for_parent))
    parent_score=parent_metric(predicted_text,golden_labels,graph_for_parent)
    print(f'{parent_score=}')
    score_to_print.append(parent_score)
    
    return score_to_print


def main(args):

    # Access the argument values
    datapath = args.datapath
    dataprefix = args.dataset
    typeKG = args.graph_kind
    model_checkpoint = args.model_checkpoint
    experiment_name = args.experiment_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    save_model = args.save_model

    hyperparams_list=[learning_rate,batch_size,epochs]


    #CUDA CHECK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device in use is ", device)

    if model_checkpoint=="led":
        print("Model selected: LED")
        model_checkpoint="allenai/led-base-16384"
    elif model_checkpoint=="bart-base":
        print("Model selected: BART-base")
        model_checkpoint="facebook/bart-base"
    elif model_checkpoint=="bart-large":
        print("Model selected: BART-large")
        model_checkpoint="facebook/bart-large"
    elif os.path.exists(model_checkpoint):
        print(f"Model checkpoint selected from {model_checkpoint} ")
    else:
        print("Model checkpoint is not valid")
        exit()
       



    train_file = datapath +'/' + dataprefix + '_train' + '.json'
    dev_file = datapath +'/'+ dataprefix + '_dev' + '.json'
    test_file = datapath +'/' + dataprefix + '_test'+ '.json'


    print("Loading dataset from ",datapath)
    dataset = load_dataset('json', data_files={'train': train_file, 'valid': dev_file, 'test': test_file})
    
    todrop=list(set(dataset['test'].column_names)-set([typeKG,'story'])) #This line returns a list of all the columns to drop (all columns minus the ones we need (input typeKG and story))
    
    #We need to add the references for the evaluation in parent score (a table-to-text generation metric )
    graph_for_parent=dataset['test']['Instances_KG']


    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_eos_token=True)

    print("\nProcessing Dataset")
    #the processing of the data is done batches for make it faster,number of processes 4
    tokenized_dataset = dataset.map(lambda example: process_data_BART(example, tokenizer,max_input,max_target,typeKG), batched=True, num_proc=4,remove_columns=todrop)

    print("\nLoading MODEL")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.to(device)

    print("Collator for batches")
    collator = DataCollatorForSeq2Seq(tokenizer, model=model) #this is necessary for diving in batch for training

    print('Loading rouge')
    rouge = evaluate.load('rouge')


    def compute_rouge(pred): 
        predictions, labels = pred
        #decode the predictions
        decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #decode labels
        decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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
        predict_with_generate=True, #since we use validation (bc during validation we generate and compare to gold ) - backprpop error on rouge
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



    print("Training TIME..")
    start_time = time.time()
    trainer.train()

    end_time = time.time()
    training_duration = {'time(s)': end_time - start_time}
    print("tranining time was:",training_duration)

    if save_model:
        print("Saving model")
        trainer.save_model(experiment_name+"/saved_model")


    print("\nPREDICTING..")
    preds, labels, metrics = trainer.predict(tokenized_dataset['test'], num_beams=5, min_length=50, max_length=max_target, no_repeat_ngram_size=2, early_stopping=True)

    predicted_text,golden_labels=tokenize_for_evaluation(tokenizer,preds,labels)

    os.makedirs(experiment_name, exist_ok=True)

    info=[{'typeKG':typeKG},{'hyper_Params':hyperparams_list}]

    score_to_print = eval_pipeline(predicted_text,golden_labels,metrics,graph_for_parent)

    gpuUSED={'gpu':print_gpu_utilization()}#this has a print instruction alredy
    score_to_print.append(training_duration)
    score_to_print.append(gpuUSED)

    score_to_print = info + score_to_print

    outpath=experiment_name+'/'

    print(f'Writing  score report in {outpath}output_metrics.json')
        
    write_scores_outputfile_json(outpath,score_to_print)
    
    print(f"Writing predicted text in {outpath}stories.json")
    write_predictions_andGraph(outpath,predicted_text,golden_labels,dataset['test'][typeKG])



    print("DONE")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune model for content planning')
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
