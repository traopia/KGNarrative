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
from transformers import pipeline

# IMPORT DATAFRAME run with each of the file here "KGNarrative2\Datasets\withSummaryNotFinal"

#train = pd.read_json("KGNarrative2/Datasets/withSummaryNotFinal/train_summary.json")
#train = train.head()

# FUNCTIONS FOR MINING

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


# ACTION TIME

def main(argv, argc):

    
    for d in ["train","test","dev"]:
        data = pd.read_json(f"./KGNarrative2/Datasets/WebNLG/57_triples/oneClass/Trattini/{d}_57_oneClass.json")

        #data=data.head(4)#REMOVE THIS LINE
        
        df3 = mining_entites(data)
        df3["mined_kg_entities"] = df3["mined_kg_entities"] # .apply(lambda x: x[1:-1]) <= fixed a problem, if create problems add it again
        df3 = column_extracting_triples(df3)
        df3 = extract_triples_from_tuples(df3)
        df3 = get_final_kg(df3)
        df3 = df3.drop(df3.index[0])
        df3['entities_list'] = df3['triple_column'].apply(extract_entities)
        #get_csv_with_mined_semantic(df3, "./train_complete.csv")  # here the path where to save
        #to_json_format("./train_complete.json", "./train_complete.csv")
        dump_json_with_mined_semantic(df3, f"./KGNarrative2/Datasets/WebNLG/57_triples/oneClass/Trattini/oneClass_{d}.json")

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
