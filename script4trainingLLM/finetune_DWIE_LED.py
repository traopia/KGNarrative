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

max_target = 512
max_input = 512

def main(argv, arc):

    if arc!=6:
        print(" ARGUMENT USAGE IS WRONG, RUN FILE LIKE: finetune_DWIE_LED.py [datapath] [dataset] [graph_kind] [model checkpoint (folder)] [Experiment_name]")
        exit()

    dataset = argv[2]
    datapath = argv[1] 
    typeKG = argv[3]
    model_checkpoint=argv[4]
    experiment_name=argv[5]

    if model_checkpoint=="led":
        print("Model selected: LED")
        model_checkpoint="allenai/led-base-16384"
    elif model_checkpoint=="bart":
        print("Model selected: BART")
        model_checkpoint="facebook/bart-base"
    else:
        print("Model checkpoint is not valid")
        exit()


    #CUDA CHECK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device in use is ", device)


    train_file = datapath +'/'+'train.json'
    dev_file = datapath +'/' + 'validation.json'
    test_file = datapath +'/'  + 'test.json'

    print("Loading dataset from",train_file)
    dataset = load_dataset('json', data_files={'train': train_file, 'valid': dev_file, 'test': test_file})
    
    todrop=list(set(dataset['test'].column_names)-set([typeKG,'story'])) #This line returns a list of all the columns to drop (all columns minus the ones we need (input typeKG and story))
  
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_eos_token=True)

    print("\nProcessing Dataset")
    #the processing of the data is done batches for make it faster,number of processes 4
    tokenized_dataset = dataset.map(lambda example: process_data_BART(example, tokenizer,max_input,max_target,typeKG), batched=True, num_proc=4,remove_columns=todrop)

    print("\nLoading MODEL")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.config.num_beams = 2
    model.config.max_length = 2048
    model.config.min_length = 100
    model.config.length_penalty = 1.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
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
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size= 3,
        gradient_accumulation_steps=3, #compute gradient on 2 examples KG story 
        weight_decay=0.01, #regularization
        save_total_limit=1, #this is the max amount of checkpoint saved, after which previous checpoints are removed
        num_train_epochs=3,
        predict_with_generate=True, #since we use validation (bc during validation we generate and compare to gold ) - backprpop error on rouge
        generation_max_length = 2048, #max number of tokens per generation 
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

    print("Saving model")
    trainer.save_model(experiment_name+"/finetunedDWIE_BART")


    print("\nPREDICTING..")
    preds, labels, metrics = trainer.predict(tokenized_dataset['test'], num_beams=5, min_length=50, max_length=2048, no_repeat_ngram_size=2, early_stopping=True)

    predicted_text,golden_labels=tokenize_for_evaluation(tokenizer,preds,labels)

    

    print("\nRESULT SCORES:")

    bertscore = evaluate.load("bertscore")
    results_bert = bertscore.compute(predictions=predicted_text, references=golden_labels, model_type="distilbert-base-uncased")
    results_bert={i:np.mean(results_bert[i]) for i in list(results_bert.keys())[:-1]}#this line is bc there is an hashvalue in results_bert that we dont need thus we only take first 3 elemnts of dictionary and avg 

    print(f'{results_bert=}')



    bleu = evaluate.load("bleu")
    result_bleu= bleu.compute(predictions=predicted_text, references=golden_labels)
    print(f'{result_bleu=}')

    google_bleu = evaluate.load("google_bleu")
    result_google_bleu = google_bleu.compute(predictions=predicted_text, references=golden_labels)
    print(f'{result_google_bleu=}')

    scores = metrics.items()
    print(f'Results: {scores}')

    gpuUSED={'gpu':print_gpu_utilization()}#this has a print instruction alredy

    outpath=experiment_name+'/'
    print(f'Writing  score report in {outpath}output_metrics.txt')
    score_to_print=[metrics,result_bleu,result_google_bleu,results_bert,training_duration,gpuUSED]
    write_scores_outputfile(outpath,score_to_print)

    print(f"Writing predicted text in {outpath}stories.json")
    write_predictions(outpath,predicted_text,golden_labels)


    print("DONE")




if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
