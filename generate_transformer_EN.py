
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import os
import nltk
import torch
import evaluate
import sys
from torch.utils.data import Dataset, DataLoader
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
nltk.download('punkt')


def check_gpu_availability():
    # Check if CUDA is available
    print(f"Cuda is available: {torch.cuda.is_available()}")

def getting_device(gpu_prefence=True) -> torch.device:
    """
    This function gets the torch device to be used for computations, 
    based on the GPU preference specified by the user.
    """
    
    # If GPU is preferred and available, set device to CUDA
    if gpu_prefence and torch.cuda.is_available():
        device = torch.device('cuda')
    # If GPU is not preferred or not available, set device to CPU
    else: 
        device = torch.device("cpu")
    
    # Print the selected device
    print(f"Selected device: {device}")
    
    # Return the device
    return device

# Define a function to print GPU memory utilization
def print_gpu_utilization():
    # Initialize the PyNVML library
    nvmlInit()
    # Get a handle to the first GPU in the system
    handle = nvmlDeviceGetHandleByIndex(0)
    # Get information about the memory usage on the GPU
    info = nvmlDeviceGetMemoryInfo(handle)
    # Print the GPU memory usage in MB
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# Define a function to print training summary information
def print_summary(result):
    # Print the total training time in seconds
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    # Print the number of training samples processed per second
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # Print the GPU memory utilization
    print_gpu_utilization()






def main(argv, arc):
    check_gpu_availability()
    device = getting_device(gpu_prefence=True)
    print(device)

    if arc!=3:
        print(" ARGUMENT USAGE IS WRONG, RUN FILE LIKE: generate_transformer_DWIE.py Instances Instances")
        exit()

    how = argv[2]+" "
    experiment_name = argv[1] 
    method = 'EN'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    print(how)
    print(experiment_name)
    train_file = 'EventNarrative/' + experiment_name + '/' + method +'_train.json'
    dev_file = 'EventNarrative/' +  experiment_name + '/' + method + '_validation.json'
    test_file = 'EventNarrative/' + experiment_name + '/' + method + '_test.json'

    dataset = load_dataset('json', data_files={'train': train_file, 'valid': dev_file, 'test': test_file})

    ### let's check the data

    print(len(dataset['valid'][0]['story']))
    print(dataset['test'][0][f'{how}Knowledge Graph'])

    print(dataset['test'][0][f'{how}Knowledge Graph'])



    ### data looks OK, let's continue with the tokenizer
    #max_input = 1000 #number of tokens we expect -DEPENDS ON size of inputs (for the tensor) - sentence and then it pads the rest - 
    #such that tensors equal in shape (no pads when training bc we alreasy have same shape) - batches of same size

    #max_target = np.max([len(nltk.word_tokenize(dataset['test'][i]['story'])) for i in range(100)])+200
    max_target = 512

    #max_input = int(np.max([len(nltk.word_tokenize(dataset['test'][i][f'{how}Knowledge Graph'])) for i in range(100)])+200)

    max_input = 512
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    print('max_target',max_target)
    print('max_input',max_input)
    print(np.max([len(nltk.word_tokenize(dataset['valid'][i]['story'])) for i in range(100)]))
    print(np.max([len(nltk.word_tokenize(dataset['train'][i]['story'])) for i in range(600)]))
    print(np.max([len(nltk.word_tokenize(dataset['valid'][i][f'{how}Knowledge Graph'])) for i in range(100)]))
    print(np.max([len(nltk.word_tokenize(dataset['train'][i][f'{how}Knowledge Graph'])) for i in range(600)]))

    def process_data_to_model_inputs(data_to_process):
        #get the dialogue text
        inputs = [graph for graph in data_to_process[f'{how}Knowledge Graph']]
        #tokenize text
        model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)

        #tokenize labels
        #with tokenizer.as_target_tokenizer():
        targets = [target for target in data_to_process['story']]
        model_targets = tokenizer(targets, max_length=max_target, padding='max_length', truncation=True)
        
        model_inputs['labels'] = model_targets['input_ids']
        #reuturns input_ids, attention_masks, labels
        

        data_to_process["input_ids"] = model_inputs.input_ids
        data_to_process["attention_mask"] = model_inputs.attention_mask

        # create 0 global_attention_mask lists
        data_to_process["global_attention_mask"] = len(data_to_process["input_ids"]) * [
            [0 for _ in range(len(data_to_process["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        data_to_process["global_attention_mask"][0][0] = 1
        data_to_process["labels"] = model_targets.input_ids

        # We have to make sure that the PAD token is ignored
        data_to_process["labels"] = [
            [0 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in data_to_process["labels"]
        ]

        return data_to_process



    ### let's drop the columns that we won't need
    tokenize_data = dataset.map(process_data_to_model_inputs, batched = True, remove_columns=['Instance Knowledge Graph', 'story'])
    #my_dataloader = DataLoader(tokenize_data, batch_size=1)


    #train = [input_ids.to(device),attention_mask.to(device) ,labels.to(device) for input_ids, attention_mask, labels in tokenize_data['train']]
    #print(train.is_cuda)
    #print(tokenize_data['train'])
    ### load model
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False).to(device)

    #collator to create batches. It preprocess data with the given tokenizer
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

    # set generate hyperparameters
    model.config.num_beams = 2
    model.config.max_length = int(max_target) #300
    model.config.min_length =  int(max_target)-100#120
    model.config.length_penalty = 2
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    #####################
    # metrics
    # compute rouge for evaluation 
    #####################
    metric = load_metric('rouge')

    def compute_rouge(pred):
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

    training_args = transformers.Seq2SeqTrainingArguments(
        'output',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size= 6,
        gradient_accumulation_steps=2, #compute gradient on 2 examples KG story 
        weight_decay=0.01, #regularization 
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True, #since we use validation (bc during validation we generate and compare to gold ) - backprpop error on rouge
        generation_max_length = int(max_target), #max number of tokens per generation 
        generation_num_beams=5, #decoding strategy! greedy search, beam search 
        eval_accumulation_steps=1, #backprop  
        fp16=False #memory management 
        )

    ### almost training time
    trainer = transformers.Seq2SeqTrainer(
        model, 
        args = training_args,
        train_dataset=tokenize_data['train'],
        eval_dataset=tokenize_data['valid'],
        data_collator=collator,
        compute_metrics=compute_rouge
    )

    ### check GPU


    ### training time - takes around 15 mins

    trainer.train()
    ### save the model for later use
    #os.mkdir(Working_Dir + '/generated_text/by_BART/' + experiment_name + how )
    trainer.save_model("LONGFormer_model_params" + experiment_name + how )

    ### let's have a look at the predictions

    preds, labels, metrics = trainer.predict(tokenize_data['test'], num_beams=5, min_length=max_input, max_length=max_input+50, no_repeat_ngram_size=2, early_stopping=True)

    pred = []
    lab = []
    for gen, gold in zip(preds, labels):
        gen = tokenizer.decode(gen, skip_special_tokens=True)
        gen = str(gen)
        gen = gen
        pred.append(gen)
        #print(pred)
        #print(f'Generated text: {gen}')
        gold = tokenizer.decode(gold, skip_special_tokens=True)
        gold = str(gold)
        gold = [gold]
        lab.append(gold)
        #results = bleu.compute(predictions=gen, references=gold)
        #print(results)
        #print(f'Reference text: {gold}')
    print(len(pred))  
    print(len(lab))

    bleu = evaluate.load("bleu")
    result_bleu= bleu.compute(predictions=pred, references=lab)
    print(result_bleu)

    google_bleu = evaluate.load("google_bleu")
    result_google_bleu = google_bleu.compute(predictions=pred, references=lab)
    print(result_google_bleu)


    #os.mkdir( './generated_text/by_LONG/' + experiment_name + how )
    #outfile_path =  './generated_text/by_LONG/' + experiment_name + how + '/generated_text_with_refs_bleu.txt'

    outfile = open(how + '_long_output_metrics_params.txt', "a", encoding='utf-8')

    scores = metrics.items()
    print(f'Results: {scores}')
    print(f'Bleu Results: {result_bleu}')
    print(f'Google Bleu Results: {result_google_bleu}')

    outfile.write(f'Test Results: {scores}\n\n')
    outfile.write(f'Bleu Results: {result_bleu}\n\n')
    outfile.write(f'Google Bleu Results: {result_google_bleu}\n\n')
    outfile.write(f'Metrics: {metrics}\n\n')

    outfile.close()

    outfile1 = open(how + '_stories.json', "a", encoding='utf-8')

    outfile1.write('[')

    for gen, gold in zip(preds, labels):
        gen = tokenizer.decode(gen, skip_special_tokens=True)
        #print(f'Generated text: {gen}')
        outfile1.write('{')
        outfile1.write(f'"Generated text": " {gen} "\n')
        outfile1.write(',')
        gold = tokenizer.decode(gold, skip_special_tokens=True)
        #print(f'Reference text: {gold}')
        outfile1.write(f'"Reference text": " {gold} " \n\n')
        outfile1.write('}')
        outfile1.write(',')

    outfile1.write('{}]')
    outfile1.close()


if __name__ == '__main__':
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(5890 * 100+ 10)
    main(sys.argv, len(sys.argv))
