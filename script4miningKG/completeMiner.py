print("IMPORTING")

# INSTALLING
'''
!pip
install
inltk
!pip
install
transformers
!pip
install
datasets
!pip
install
pynvml
!pip
install
evaluate
!pip
install
sentencepiece
!pip
install
flair
'''

# ===========================================
# ||                                       ||
# ||Section 1: Importing modules           ||
# ||                                       ||
# ===========================================

# IMPORTING
print("IMPORTING")
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import os
import nltk
import torch
import evaluate
import sys
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from sklearn.model_selection import train_test_split
import torch.cuda as cuda
import gc
from flair.models import SequenceTagger
from flair.data import Sentence
import csv, json
from transformers import pipeline

# IMPORT DATAFRAME run with each of the file here "KGNarrative2\Datasets\withSummaryNotFinal"

#train = pd.read_json("KGNarrative2/Datasets/withSummaryNotFinal/train_summary.json")
#train = train.head()

# ===========================================
# ||                                       ||
# ||Section 2: Functions => Utils GPU      ||
# ||                                       ||
# ===========================================


# UTILS FOR GPU 
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

def clean_gpu():
    # Get current GPU memory usage
    print("BEFORE CLEANING:")
    print(f"Allocated: {cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {cuda.memory_cached() / 1024 ** 3:.2f} GB")
    print("\n")
    # Free up PyTorch and CUDA memory
    torch.cuda.empty_cache()
    cuda.empty_cache()
    
    # Run garbage collection to free up other memory
    gc.collect()
    
    # Get new GPU memory usage
    print("AFTER CLEANING:")
    print(f"Allocated: {cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {cuda.memory_cached() / 1024 ** 3:.2f} GB")


# ===========================================
# ||                                       ||
# ||Section 3: Functions => Miner          ||
# ||                                       ||
# ===========================================

# FUNCTIONS FOR MINING

def mining_type_of_news(df,model_name ='facebook/bart-large-mnli'):

  # import pipeline on gpu
  classifier = pipeline("zero-shot-classification", model = model_name, device = 0)

  # type of news 
  candidate_labels = ["Tech", "Entertainment", "Sport", "Business", "Politics"]  
  
  # Create a list to hold the predicted labels
  predicted_type_of_news= []

  for story in df["story"]:

      # saving the story we need to classify
      sequence_to_classify = story
      
      # using NLI to classify
      prediction = classifier(sequence_to_classify, candidate_labels)

      # accessing the main label 
      predicted_label = prediction["labels"][0]
      # Add the predicted label to the list
      predicted_type_of_news.append(predicted_label)

  # adding the mined info as column of our data
  df["predicted_label1"] = predicted_type_of_news
  
  return df

def mining_summary(df,model_name = "deep-learning-analytics/automatic-title-generation"): # old one =>"google/pegasus-multi_news"
  
  # Getting Device
  device = getting_device(gpu_prefence=True)

  # Load the tokenizer and model from Hugging Face
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # load the model
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

  # Create a list to hold the summaries
  summaries = []

  # Loop through the stories
  for story in df["story"]:

      # Tokenize the story
      inputs = tokenizer.encode(story, return_tensors="pt", max_length=1024, truncation=True).to(device)

      # Generate the summary
      outputs = model.generate(inputs, max_length=30, min_length=1, length_penalty=15.0, num_beams=4, early_stopping=True)
      
      # Decode the summary and add it to the list
      summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

      # append to the summary list
      summaries.append(summary)
    
  # Add the summaries to the dataframe
  df["core description"] = summaries

  # REMOVE IF IT CREATES PROBLEM WITH YOUR PARTICULAR DATASET
  # df["core description"] = df["core description"].apply(lambda x: x[1:])

  return df


def extract_named_unique_entities_with_filters(text, ner_name='ner-ontonotes-large', pos_name="pos-fast"):
    # Load the NER model
    taggerNer = SequenceTagger.load(ner_name)
    taggerPos = SequenceTagger.load(pos_name)
    sentence = Sentence(text)
    taggerNer.predict(sentence,verbose=False)
    taggerPos.predict(sentence,verbose=False)
    entities = []
    for entity in sentence.get_spans('ner'):
        entity_text = entity.text
        entity_type = entity.labels[0].value
        tokens = entity.tokens
        pos_label = [token.get_labels()[0].value for token in tokens][0]

        # FILTERING OUT ADJECTIVES
        if pos_label != "JJ" and pos_label != "JJR":
            if entity_type != "PERCENT" and entity_type != "QUANTITY" and entity_type != "CARDINAL":
                entities.append((entity.text, entity.labels[0].value))
    entities = tuple(set(entities))
    output = ''
    for entity in entities:
        output += entity[0] + ' - type - ' + entity[1] + ' | '
    output = output[:-3] + ''
    return output


def mining_entites(df):
    # apply to dataframe
    df['mined_kg_entities'] = df['story'].apply(extract_named_unique_entities_with_filters)

    return df


def check_it_function(dataframe, index):
    for j, i in enumerate(dataframe.iloc[index]):
        print(f"{dataframe.columns[j]}", "=>", i, "\n")


def extract_triples(text):
    # Split the string into individual triples
    triples = [t.strip() for t in text.split('|')]
    # Split each triple into its constituent parts
    triples = [tuple(t.split('- type - ')) for t in triples]
    return triples


def column_extracting_triples(df):
    df['triple_column'] = df['mined_kg_entities'].apply(extract_triples)

    # REMOVING ORDINAL AND LANGUAGE
    df['triple_column'] = [[t for t in row if len(t) > 1 if t[1] not in ['ORDINAL', 'LANGUAGE']] for row in
                           df['triple_column']]
    return df


def extract_triples_from_tuples(df):
    new_triples = []
    for row in df['triple_column']:
        new_row = []
        for triple in row:
            if triple[1] == 'WORK_OF_ART':
                new_row.append(('news', 'has item', triple[0]))
                new_row.append((triple[0], 'type', 'work of art'))
            elif triple[1] == 'LAW':
                new_row.append(('news', 'has item', triple[0]))
                new_row.append((triple[0], 'type', 'law'))
            elif triple[1] == 'FAC':
                new_row.append(('news', 'has item', triple[0]))
                new_row.append((triple[0], 'type', 'facility'))
            elif triple[1] == 'MONEY':
                new_row.append(('news', 'has item', triple[0]))
                new_row.append((triple[0], 'type', 'money'))
            elif triple[1] == 'PRODUCT':
                new_row.append(('news', 'has item', triple[0]))
                new_row.append((triple[0], 'type', 'product'))
            elif triple[1] == 'TIME':
                new_row.append(('news', 'has time', triple[0]))
                new_row.append((triple[0], 'type', 'time'))
            elif triple[1] == 'DATE':
                new_row.append(('news', 'has time', triple[0]))
                new_row.append((triple[0], 'type', 'time'))
            elif triple[1] == 'LOC':
                new_row.append(('news', 'has place', triple[0]))
                new_row.append((triple[0], 'type', 'place'))
            elif triple[1] == 'EVENT':
                new_row.append(('news', 'has event', triple[0]))
                new_row.append((triple[0], 'type', 'event'))
            elif triple[1] == 'PERSON':
                new_row.append(('news', 'has actor', triple[0]))
                new_row.append((triple[0], 'type', 'person'))
            elif triple[1] == 'NORP':
                new_row.append(('news', 'has actor', triple[0]))
                new_row.append((triple[0], 'type', 'Nationalities or Religious or Political Groups'))
            elif triple[1] == 'ORG':
                new_row.append(('news', 'has actor', triple[0]))
                new_row.append((triple[0], 'type', 'organization'))
            elif triple[1] == 'GEO':
                new_row.append(('news', 'has actor', triple[0]))
                new_row.append((triple[0], 'type', 'Geo-Political Entity'))
        new_triples.append(new_row)

    df['new_triples'] = new_triples
    df['final_triples'] = ["{" + " | ".join([f"{triple[0]} - {triple[1]} - {triple[2]}" for triple in row]) + "}" for
                           row in df['new_triples']]
    return df


def get_final_kg(df):
    df["final_triples"] = df["final_triples"].apply(lambda x: x[1:-1])
    df["semantic_of_news"] = "news - type - " + df["predicted_label1"] + " | " + df["final_triples"] # <------ THIS IS THE GOOD ONE BUT THERE IS NO LABEL
    #df["semantic_of_news"] =  df["final_triples"]


    return df

def extract_entities(triples_column):
    entities_list = [triple[0] for triple in triples_column]
    return entities_list


def get_csv_with_mined_semantic(df, path):
    df.drop(['predicted_label1',
        'core_description', 'mined_kg_entities', 'triple_column', 'new_triples',
        'final_triples'], axis = 1, inplace = True) 
  
    df.to_csv(path, index=False)

def dump_json_with_mined_semantic(df, path):
    df.drop(['predicted_label1', 'mined_kg_entities', 'triple_column', 'new_triples', #Here there was also core_description but not sure i wanna dumpit
        'final_triples'], axis = 1, inplace = True) 
    with open(path, "w") as f:
            json.dump(df.to_dict('records'), f, indent=4)
  
    


def to_json_format(json_filename, csv_filename):
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        columns = next(reader)
    with open(json_filename, "w") as jsonfile:
        jsonfile.write('[')
        for row in csv.DictReader(open(csv_filename), fieldnames=columns):
            json.dump(row, jsonfile, indent=4)
            jsonfile.write(',')
            jsonfile.write('\n')
        jsonfile.write('{}')
        jsonfile.write(']')


# ===========================================
# ||                                       ||
# ||Section 4: Main                        ||
# ||                                       ||
# ===========================================

def wrap(filename):
    #data = pd.read_json(f"./KGNarrative2/Datasets/WebNLG/57_triples/oneClass/Trattini/{d}_57_oneClass.json")
    data = pd.read_json(filename)
    #data=data.head(4)#REMOVE THIS LINE
    print("data loaded from json" + filename)
    df1 = mining_type_of_news(data)
    clean_gpu()
    df2 = mining_summary(df1)
    clean_gpu()
    df3 = mining_entites(df2)
    df3["mined_kg_entities"] = df3["mined_kg_entities"] # .apply(lambda x: x[1:-1]) <= fixed a problem, if create problems add it again
    df3 = column_extracting_triples(df3)
    df3 = extract_triples_from_tuples(df3)
    df3 = get_final_kg(df3)
    df3 = df3.drop(df3.index[0])
    df3['entities_list'] = df3['triple_column'].apply(extract_entities)
    #get_csv_with_mined_semantic(df3, "./train_complete.csv")  # here the path where to save
    #to_json_format("./train_complete.json", "./train_complete.csv")
    #dump_json_with_mined_semantic(df3, f"./KGNarrative2/Datasets/WebNLG/57_triples/oneClass/Trattini/oneClass_{d}.json")
    dump_json_with_mined_semantic(df3, filename)
    print('done with ', filename)

def main(argv, argc):

    # CHECK IF GPU IS UP
    check_gpu_availability()
    print("STARTING..")

    # SAVE THE DEVICE WE ARE WORKING WITH
    device = getting_device(gpu_prefence=True)
    if argc != 2:
        print("Usage: python3 main.py path")
        raise Exception("Usage: python3 main.py path")
    path=argv[1]
    if not os.path.isdir(path): 
        print("The path doesn't exist")
        raise Exception("The path doesn't exist")

    for d in ["train","test","validation"]:

        filename=f'{path}/EN_{d}.json'

        print("Working on ", filename)
        wrap(filename)

        # #data = pd.read_json(f"./KGNarrative2/Datasets/WebNLG/57_triples/oneClass/Trattini/{d}_57_oneClass.json")
        # data = pd.read_json(f'Datasets/DWIE/DWIE_cleaned/{d}_cleaned.json')
        # #data=data.head(4)#REMOVE THIS LINE
        # df1 = mining_type_of_news(data)
        # clean_gpu()
        # df2 = mining_summary(df1)
        # clean_gpu()
        # df3 = mining_entites(df2)
        # df3["mined_kg_entities"] = df3["mined_kg_entities"] # .apply(lambda x: x[1:-1]) <= fixed a problem, if create problems add it again
        # df3 = column_extracting_triples(df3)
        # df3 = extract_triples_from_tuples(df3)
        # df3 = get_final_kg(df3)
        # df3 = df3.drop(df3.index[0])
        # df3['entities_list'] = df3['triple_column'].apply(extract_entities)
        # #get_csv_with_mined_semantic(df3, "./train_complete.csv")  # here the path where to save
        # #to_json_format("./train_complete.json", "./train_complete.csv")
        # #dump_json_with_mined_semantic(df3, f"./KGNarrative2/Datasets/WebNLG/57_triples/oneClass/Trattini/oneClass_{d}.json")
        # dump_json_with_mined_semantic(df3, f'Datasets/DWIE/DWIE_cleaned/{d}_cleaned_mined.json')

if __name__ == '__main__':
    print("BEFORE MAIN")
    main(sys.argv, len(sys.argv))
