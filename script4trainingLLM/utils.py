from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import json
from parent import parent
import numpy as np


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



def process_data_LED(data_to_process,tokenizer,max_input,max_target,typeKG ):

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


    # create 0 global_attention_mask lists
    data_to_process["global_attention_mask"] = len(data_to_process["input_ids"]) * [
        [0 for _ in range(len(data_to_process["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    data_to_process["global_attention_mask"][0][0] = 1

    return data_to_process


def process_data_to_model_inputs_OLD(data_to_process,tokenizer,max_input,max_target,how):
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

def write_scores_outputfile(outpath,score_list):
    outfile = open(outpath + 'output_metrics.txt', "a", encoding='utf-8')
    for s in score_list:
        outfile.write(f'{s}\n\n')

    outfile.close()

def write_predictions(outpath,preds,labels):#OBSOLETE

    outfile = open(outpath + 'stories.json', "a", encoding='utf-8')

    outfile.write('[')

    for gen, gold in zip(preds, labels):
        #print(f'Generated text: {gen}')
        outfile.write('{')
        outfile.write(f'"Generated_text":"{gen}"\n')
        outfile.write(',')
        outfile.write(f'"Reference_text":"{gold}"\n\n')
        outfile.write('}')
        outfile.write(',')

    outfile.write(']')
    outfile.close()

def write_predictions_andGraph(outpath,preds,labels,test_data):

    output=[]
    for graph,label,generated in zip(test_data,labels,preds):
        output.append({'graph':graph,'target':label,'generated_story':generated})
    
    with open(outpath + 'stories_withInput.json', 'w') as outfile:
        json.dump(output, outfile,indent=4)
      


# Define a function to print GPU memory utilization
def print_gpu_utilization():
    # Initialize the PyNVML library
    nvmlInit()
    # Get a handle to the first GPU in the system
    handle = nvmlDeviceGetHandleByIndex(0)
    # Get information about the memory usage on the GPU
    info = nvmlDeviceGetMemoryInfo(handle)
    # Print the GPU memory usage in MB
    gpuused=info.used//1024**2
    print(f"GPU memory occupied: {gpuused} MB.")
    return gpuused

# Define a function to print training summary information
def print_summary(result):
    # Print the total training time in seconds
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    # Print the number of training samples processed per second
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # Print the GPU memory utilization
    print_gpu_utilization()


def convert_graph_to_table(graph):
    table = []
    for i in range(len(graph)):
        row = []
        for j in range(len(graph[i])):
            row.append(graph[i][j])
        table.append(row)
    return table




def parent_metric(predictions,references,tables):
    """This functions calculate the parent metric with the implementation from https://github.com/KaijuML/parent
    Inputs are just list of strings. They then get converted according to the format required in the module.
    """
    predictions=[p.split() for p in predictions]
    references=[r.split() for r in references]
    tables =[[[p.split() for p in x.split(" - ")] for x in t.strip().split(" | ")] for t in tables]#THIS makes a list of lists of lists from triples. Removing the last element because it is hasCore.

    #SANITY CHECK
    for x in tables:
        for t in x:
            if len(t)!=3:
                raise Exception("Triple is not 3 elements long. PARENT METRIC ERROR FORMAT: \n",t)

    precision, recall, f_score = parent(
    predictions,
    references,
    tables,
    avg_results=True,
    n_jobs=1,
    use_tqdm='notebook'
    )
    return {"PARENT":{'precision':precision,'recall':recall,'f_score':f_score}}