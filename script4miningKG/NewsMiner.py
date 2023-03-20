# INSTALLING
'''
!pip install transformers
!pip install datasets
!pip install pynvml
!pip install evaluate
!pip install sentencepiece
!pip install flair
'''

# ===========================================
# ||                                       ||
# ||Section 1: Importing modules           ||
# ||                                       ||
# ===========================================

# IMPORTING
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForSeq2SeqLM
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
import sys


# ===========================================
# ||                                       ||
# ||Section 2: Functions => Utils GPU      ||
# ||                                       ||
# ===========================================


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
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


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

# 1 => FUNCTION FOR MINING NEWS TYPE

def mining_type_of_news(df, model_name="abhishek/autonlp-bbc-news-classification-37229289"):
    # Getting Device
    device = getting_device(gpu_prefence=True)

    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # Load the configuration of the model
    config = AutoConfig.from_pretrained(model_name)
    labels = config.label2id.keys()

    # Create a list to hold the predicted labels
    predicted_type_of_news = []

    # Loop over each story in the DataFrame
    max_length = tokenizer.model_max_length

    # TODO: make it better (bc max input of those models is 512)
    for story in df["story"]:
        # Encode the story text using the tokenizer
        inputs = tokenizer.encode(story, max_length=max_length, return_offsets_mapping=False, stride=0,
                                  return_tensors='pt').to(device)

        # Make a prediction using the model
        outputs = model(inputs)

        # Get the predicted label
        predicted_label_id = outputs.logits.argmax().item()

        predicted_label = list(labels)[predicted_label_id]

        # Add the predicted label to the list
        predicted_type_of_news.append(predicted_label)

    # adding the mined info as column of our data
    df["predicted_label1"] = predicted_type_of_news

    return df


# 2 => FUNCTION 4 MINING SUMMARIES

def mining_summary(df, model_name="google/pegasus-multi_news"):
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
        outputs = model.generate(inputs, max_length=30, min_length=1, length_penalty=15.0, num_beams=4,
                                 early_stopping=True)

        # Decode the summary and add it to the list
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # append to the summary list
        summaries.append(summary)

    # Add the summaries to the dataframe
    df["core description"] = summaries

    # REMOVE IF IT CREATES PROBLEM WITH YOUR PARTICULAR DATASET
    df["core description"] = df["core description"].apply(lambda x: x[1:])

    return df


# 3 => FUNCTION FOR MINING NEWS CORE DESCRIPTION

# Getting the right format and unique entities
def extract_named_unique_entities_with_filters(text,ner_name = "ner-ontonotes-fast", pos_name = "pos-fast"):
    # Load the NER model
    taggerNer = SequenceTagger.load(ner_name)
    taggerPos = SequenceTagger.load(pos_name)
    sentence = Sentence(text)
    taggerNer.predict(sentence)
    taggerPos.predict(sentence)
    entities = []
    for entity in sentence.get_spans('ner'):
        entity_text = entity.text
        entity_type = entity.labels[0].value
        tokens = entity.tokens
        pos_label = [token.get_labels()[0].value for token in tokens][0]

        # FILTERING OUT ADJECTIVES
        if pos_label != "JJ" and pos_label != "JJR":
          if entity_type != "PERCENT" and entity_type != "QUANTITY" and entity_type != "CARDINAL":
            entities.append((entity.text,entity.labels[0].value))
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

# ===========================================
# ||                                       ||
# ||Section 4: Functions => KG composer    ||
# ||                                       ||
# ===========================================

# UTILS 4 PRINTING THE DATAFRAME INFORMATION

def check_it_function(dataframe, index):
    for j, i in enumerate(dataframe.iloc[index]):
        print(f"{dataframe.columns[j]}", "=>", i, "\n")

# EXTRACT TRIPLES

def extract_triples(text):
    # Split the string into individual triples
    triples = [t.strip() for t in text.split('|')]
    # Split each triple into its constituent parts
    triples = [tuple(t.split('- type - ')) for t in triples]
    return triples

def column_extracting_triples(df):
  df['triple_column'] = df['mined_kg_entities'].apply(extract_triples)

  # REMOVING ORDINAL AND LANGUAGE
  df['triple_column'] = [[t for t in row if len(t) > 1 if t[1] not in ['ORDINAL', 'LANGUAGE']] for row in df['triple_column'] ]
  return df

def extract_triples_from_tuples(df):
  new_triples = []
  for row in df['triple_column']:
      new_row = []
      for triple in row:
          if triple[1] == 'WORK_OF_ART':
              new_row.append(('news', 'hasItem', triple[0]))
              new_row.append((triple[0], 'type', 'work of art'))
          elif triple[1] == 'LAW':
              new_row.append(('news', 'hasItem', triple[0]))
              new_row.append((triple[0], 'type', 'law'))
          elif triple[1] == 'FAC':
              new_row.append(('news', 'hasItem', triple[0]))
              new_row.append((triple[0], 'type', 'facility'))
          elif triple[1] == 'MONEY':
              new_row.append(('news', 'hasItem', triple[0]))
              new_row.append((triple[0], 'type', 'money'))
          elif triple[1] == 'PRODUCT':
              new_row.append(('news', 'hasItem', triple[0]))
              new_row.append((triple[0], 'type', 'product'))
          elif triple[1] == 'TIME':
              new_row.append(('news', 'hasTime', triple[0]))
              new_row.append((triple[0], 'type', 'time'))
          elif triple[1] == 'DATE':
              new_row.append(('news', 'hasTime', triple[0]))
              new_row.append((triple[0], 'type', 'time'))
          elif triple[1] == 'LOC':
              new_row.append(('news', 'hasPlace', triple[0]))
              new_row.append((triple[0], 'type', 'place'))
          elif triple[1] == 'EVENT':
              new_row.append(('news', 'hasEvent', triple[0]))
              new_row.append((triple[0], 'type', 'event'))
          elif triple[1] == 'PERSON':
              new_row.append(('news', 'hasActor', triple[0]))
              new_row.append((triple[0], 'type', 'person'))
          elif triple[1] == 'NORP':
              new_row.append(('news', 'hasActor', triple[0]))
              new_row.append((triple[0], 'type', 'Nationalities or Religious or Political Groups'))
          elif triple[1] == 'ORG':
              new_row.append(('news', 'hasActor', triple[0]))
              new_row.append((triple[0], 'type', 'organization'))
          elif triple[1] == 'GEO':
              new_row.append(('news', 'hasActor', triple[0]))
              new_row.append((triple[0], 'type', 'Geo-Political Entity'))
      new_triples.append(new_row)

  df['new_triples'] = new_triples
  df['final_triples'] = [" | ".join([f"{triple[0]} - {triple[1]} - {triple[2]}" for triple in row])  for row in df['new_triples']]
  return df

def get_final_kg(df):
  df["final_triples"] = df["final_triples"].apply(lambda x: x[1:-1])
  df["semantic_of_news_noCore"] =  " news - type - " + df["predicted_label1"] + " | "+  df["final_triples"] #+ " | news - hasCore - " + "'" + df["core description"] + "'" 
  df["semantic_of_news"] =  " news - type - " + df["predicted_label1"] + " | "+  df["final_triples"] + " | news - hasCore - " + "'" + df["core description"] + "'"
  return df


# ===========================================
# ||                                       ||
# ||Section 5: Functions => Getting Files  ||
# ||                                       ||
# ===========================================

def get_csv_with_mined_semantic(df,path):
  df.drop(['Unnamed: 0','predicted_label1',
        'core description', 'mined_kg_entities', 'triple_column', 'new_triples',
        'final_triples'], axis = 1, inplace = True)
  df.to_csv(path, index=False)


def get_csv_with_mined_semantic_concatenated_kginstances(df, path):
    df.drop(['Unnamed: 0', 'predicted_label1',
             'core description', 'mined_kg_entities', 'triple_column', 'new_triples',
             'final_triples'], axis=1, inplace=True)

    df["Instance_NewsKG"] =  df["Instances Knowledge Graph"] + df["semantic_of_news"].apply(
        lambda x: x[2:-1]) 
    df["Instances_NewsKG_noCore"] =  df["Instances Knowledge Graph"] + df["semantic_of_news_noCore"].apply(
        lambda x: x[2:-1]) 
    df["Types_NewsKG_noCore"] =  df["Instance_NewsKG_noCore"] + df["Types Knowledge Graph"]
    df["Subclass_NewsKG_noCore"] =  df["Types_NewsKG_noCore"] + df["Subclass Knowledge Graph"]
    df["Subclass_NewsKG"] =  df["Instance_NewsKG"] + df["Types_NewsKG_noCore"] + df["Subclass Knowledge Graph"]

    df.drop(['Instances Knowledge Graph', 'Types Knowledge Graph','Subclass Knowledge Graph', 'semantic_of_news', 'Instance_NewsKG', 'semantic_of_news','InstancesKG+NewsKG'], axis=1, inplace=True)
    

    df.to_csv(path, index=False)

def to_json_format(json_filename, csv_filename):
    with open(csv_filename, newline='') as csvfile:
      reader = csv.reader(csvfile)
      columns = next(reader)
    with open(json_filename, "w") as jsonfile:
        jsonfile.write('[')
        for row in csv.DictReader(open(csv_filename), fieldnames=columns):
            json.dump(row, jsonfile, indent = 4)
            jsonfile.write(',')
            jsonfile.write('\n')
        jsonfile.write('{}')
        jsonfile.write(']')


# ===========================================
# ||                                       ||
# ||Section 6: Functions => Others         ||
# ||                                       ||
# ===========================================


def check_story_column(df):
    # Check if the DataFrame contains a "story" column
    if "story" not in df.columns:
        raise ValueError("DataFrame does not contain a 'story' column.")

    # Check if the "story" column contains any non-empty strings
    if df["story"].dtype == "object" and not df["story"].str.strip().str.len().any():
        raise ValueError("The 'story' column contains no non-empty strings.")


# ===========================================
# ||                                       ||
# ||Section 7: Main                        ||
# ||                                       ||
# ===========================================


def main(args):

    try:
        # Get the path to the  file from the command-line arguments
        csv_path = args[1]
        # Load the CSV file into a DataFrame using Pandas
        df = pd.read_csv(csv_path)
    except:
        # Get the path to the  file from the command-line arguments
        json_path = args[1]
        # Load the JSON file into a DataFrame using Pandas
        df = pd.read_json(json_path)

    # checking if the df satisfy the requirements
    check_story_column(df)

    # check if gpu is available
    check_gpu_availability()

    # save the device we work with
    device = getting_device(gpu_prefence=True)

    # mining the type of the news
    df = mining_type_of_news(df)

    # resetting the memory
    clean_gpu()

    # mining the summary of the news
    df = mining_summary(df)

    # resetting the memory
    clean_gpu()

    # mining the entities
    df = mining_entites(df)

    # resetting the memory
    clean_gpu()

    # extractign triples
    df = column_extracting_triples(df)
    df = extract_triples_from_tuples(df)

    # mixing in final kg
    df = get_final_kg(df)

    # getting the file
    get_csv_with_mined_semantic_concatenated_kginstances(df, "./MinedDF")


if __name__ == "__main__":
    # Pass the command-line arguments to the main function
    main(sys.argv)
