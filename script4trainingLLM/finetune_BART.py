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

#PARAMERS FOR PADDING THE INPUTS
max_target = 512
max_input = 512



def main(argv, arc):

    if arc!=6:
        print(" ARGUMENT USAGE IS WRONG, RUN FILE LIKE: finetune_bart.py [datapath] [dataset] [type] [trainmode (parallel or else)] [outputfolder/Experiment_name]")
        exit()

    dataset = argv[2]
    datapath = argv[1] 
    typeKG = argv[3]
    train_mode=argv[4]
    experiment_name=argv[5]
    model_name = 'facebook/bart-base'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device in use is ", device)

    train_file = datapath +'/'+ dataset +'_train.json'
    dev_file = datapath +'/'+  dataset + '_validation.json'
    test_file = datapath +'/'  + dataset + '_test.json'

    print("Loading dataset from",train_file)
    dataset = load_dataset('json', data_files={'train': train_file, 'valid': dev_file, 'test': test_file})


    print("\nLoading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name,add_eos_token=True,)

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

    if train_mode=="parallel":
        f=fopen('config.json','r')
        config = json.load(f)

        
        #defining training arogouments
        args = Seq2SeqTrainingArguments(
        'output',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size= 6,
        gradient_accumulation_steps=6, #compute gradient on 2 examples KG story
        weight_decay=0.01, #regularization
        save_total_limit=1, #this is the max amount of checkpoint saved, after which previous checpoints are removed
        num_train_epochs=3,
        predict_with_generate=True, #since we use validation (bc during validation we generate and compare to gold ) - backprpop error on rouge
        generation_max_length = 512, #max number of tokens per generation
        generation_num_beams=5, #decoding strategy! greedy search, beam search
        eval_accumulation_steps=1, #backpro
        fp16=True, #memory managemen
        disable_tqdm=True,
        deepspeed=config)

    else:
        print("training not in parallel\n")
        #defining training arogouments
        args = Seq2SeqTrainingArguments(
        experiment_name,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size= 6,
        gradient_accumulation_steps=6, #compute gradient on 2 examples KG story 
        weight_decay=0.01, #regularization
        save_total_limit=1, #this is the max amount of checkpoint saved, after which previous checpoints are removed
        num_train_epochs=3,
        predict_with_generate=True, #since we use validation (bc during validation we generate and compare to gold ) - backprpop error on rouge
        generation_max_length = 512, #max number of tokens per generation 
        generation_num_beams=5, #decoding strategy! greedy search, beam search 
        eval_accumulation_steps=1, #backprop  
        fp16=True, #memory management
        disable_tqdm=True)
        



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
    trainer.save_model(experiment_name+"/finetuned_BART_EventNarrative")



    print("\nPREDICTING..")
    preds, labels, metrics = trainer.predict(tokenized_dataset['test'], num_beams=5, min_length=50, max_length=450, no_repeat_ngram_size=2, early_stopping=True)

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
