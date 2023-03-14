import pandas as pd
from tqdm import tqdm
import json
import csv

def concatenate_list(lst):
    return ', '.join(lst)


def preprocess_data(path_in, path_out):
    """
    Preprocess the data to have: story and Instance_Knowledge_Graph
    """
    data = pd.read_json(path_in)
    for instance in tqdm(range(len(data))):
        if type(data['types'][instance]) != list:
            data['types'][instance]= eval(data['types'][instance])
        for key,value in data['entity_ref_dict'][instance].items():
            data['narration'][instance] = data['narration'][instance].replace(key,value)
        formatted_triples = []
        for triple in data['keep_triples'][instance]: 
            formatted_triples.append((" | ".join([" - ".join(triple) for triple in data['keep_triples'][instance]])))
            data['keep_triples'][instance] = formatted_triples[0] 

    result = pd.DataFrame()  
    result['Instance_Knowledge_Graph'] = data['keep_triples'] + ' | story - hasCore - ' + data['Event_Name'] + ' | story - type - ' +  data['types'].apply(concatenate_list).apply(lambda x: x.replace(',' , ' | story - type - '))
    result['story'] = data['narration'] 
    result.to_csv(path_out, index=False)


def preprocess_to_json(csv_filename, json_filename):
    """
    Preprocess the data to be in the json format
    """
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        columns = next(reader)

    # Read the data from the CSV file and write it to a JSON file
    with open(json_filename, "w") as jsonfile:
        jsonfile.write('[')
        for row in csv.DictReader(open(csv_filename), fieldnames=columns):
            json.dump(row, jsonfile, indent = 4)
            jsonfile.write(',')
            jsonfile.write('\n')
        jsonfile.write('{}')    
        jsonfile.write(']')  



def main():
    preprocess_data('EventNarrative/test_data.json', 'EventNarrative/test_data.csv')
    preprocess_data('EventNarrative/train_data.json', 'EventNarrative/train_data.csv')
    preprocess_data('EventNarrative/dev_data.json', 'EventNarrative/val_data.csv')
    preprocess_to_json('EventNarrative/test_data.csv', 'EventNarrative/Instances/EN_test.json')
    preprocess_to_json('EventNarrative/train_data.csv', 'EventNarrative/Instances/EN_train.json')
    preprocess_to_json('EventNarrative/val_data.csv', 'EventNarrative/Instances/EN_val.json')

if __name__ == '__main__':
    main()