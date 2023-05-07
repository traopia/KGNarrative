import json
import os
from collections import Counter
from nltk.tokenize import word_tokenize
from src_preprocessing.completeMiner import *


'''
In this script we perform the preprocessing of the DWIE dataset.
We create a json file that contains the story and the linearized KG, together with the types and the subclass of the entities.
It is assumed that the dataset has been downloaded and it is in the folder data/annos_with_content.
The dataset can be downloaded from https://github.com/klimzaporojets/DWIE. After the github repo has been cloned run the following command:
python src/dwie_download.py
'''

if not os.path.exists('Dataset/DWIE'):
        os.makedirs('Dataset/DWIE')

def create_linearized_KG(data): 
    """
    This function creates a linearized KG from the KG in the json file
    """
    concept_text = dict() #dictionary that maps the concept to the text
    #KG = []
    str = '' #string that contains the linearized KG
    for i in range(len(data['concepts'])): #for each concept in the KG we create a dictionary that maps the concept to the text
        if 'text' in data['concepts'][i]:
            concept_text[data['concepts'][i]['concept']] = data['concepts'][i]['text']
        elif 'link' in data['concepts'][i]: #if the concept is a link we map it to the link
            concept_text[data['concepts'][i]['concept']] = data['concepts'][i]['link']
        else:    #if the concept is not a link and it doesn't have text we map it to an empty string
            concept_text[data['concepts'][i]['concept']] = ''
            print(data['concepts'][i]['concept'])
            
    for i,j in zip(range(len(data['relations'])),range(len(concept_text))):
        str += concept_text[data['relations'][i]['s']]+' - '+data['relations'][i]['p']+' - '+concept_text[data['relations'][i]['o']]+' | '
        #KG.append(concept_text[data['relations'][i]['s']]+' - '+data['relations'][i]['p']+' - '+concept_text[data['relations'][i]['o']] )

    return str



def create_types_KG(data):
    ''' This function creates a linearized KG that contains the types of the entities in the KG'''
    concept_text = dict()
    str = ''
    concept = []
    for i in range(len(data['concepts'])):
        if 'text' in data['concepts'][i]:
            concept_text[data['concepts'][i]['concept']] = data['concepts'][i]['text']
        elif 'link' in data['concepts'][i]:
            concept_text[data['concepts'][i]['concept']] = data['concepts'][i]['link']
        else:    
            #print('no text for concept: ', data['concepts'][i]['concept'])
            concept_text[data['concepts'][i]['concept']] = ''
            
    for i in range(len(concept_text)):

        types = [j for j in data['concepts'][i]['tags'] if 'type' in j]
        types = [i.split("::") for i in types]

        for g in types:
            str += concept_text[i] +' - ' + g[0] + ' - ' + g[1] + ' | '
            if g[1] not in concept:
                concept.append(g[1])
    return str, concept


def create_subclass_KG(data):
    ''' This function creates a linearized KG that contains the subclass of the entities in the KG'''
    a, concept = create_types_KG(data)
    f = open("data/schema/ner.rdf", "r")
    subclass_triples = []
    for line in f:
        line_split = line.split()
        if line_split:
            if line_split[1] == 'subclass_of' :
                subclass_triples.append(line_split)

    result = ''       
    for c in concept:
        for i in subclass_triples:
            if c ==i[0]:
                result += i[0] + ' - ' + i[1] + ' - ' + i[2][:-1] + ' | '

    return result 


def create_experiment_linearized(data): 
    """
    This function creates a dictionary that contains the story and the linearized KG
    """
    dict = {}
    dict['story'] = data['content'].replace('\n', ' ')
    dict['Instances Knowledge Graph'] = create_linearized_KG(data)
    types, concepts = create_types_KG(data)
    dict['Types Knowledge Graph'] = types + create_linearized_KG(data)
    dict['Subclass Knowledge Graph'] = create_subclass_KG(data) + types + create_linearized_KG(data)
    return dict




def frequency_of_types(data):
    ''' This function returns a dictionary that contains the frequency of each type in the KG'''
    dict = {}
    for i in range(len(data['concepts'])):
        types = [j for j in data['concepts'][i]['tags'] if 'type' in j]
        for g in types:
            if g.split("::")[1] not in dict:
                dict[g.split("::")[1]] = 1
            else:
                dict[g.split("::")[1]] += 1
    return dict

def overall_frequency_of_types(directory= 'DWIE/data/annos_with_content/'):
    ''' This function returns a dictionary that contains the frequency of each type in the KG of all the files in the directory'''
    big_dict = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        with open(path) as g:
            try:
                data = json.load(g) 
                dicts = frequency_of_types(data)
                big_dict = Counter(big_dict) + Counter(dicts)

            except BaseException as e:
<<<<<<< HEAD
                print('The file contains invalid JSON')
                print(path)
=======
                print('Some files contain invalid JSON')

>>>>>>> ba1db5e (Update preprocessing_DWIE.py)
    big_dict = sorted(big_dict.items(), key=lambda x:x[1], reverse=True)
    return big_dict

def remove_long_stories(data):
    ''' Remove data with stories longer than 1024 tokens'''
    selected_data = []
    for d in data:
        if len(word_tokenize(d['story'])) < 1024:
            selected_data.append(d)       
    print(f'From len {len(data)} only {len(selected_data)} selected ')  
    return selected_data


def prepare_KG(directory,outdir):
    '''This function creates a json file that contains the linearized KG and the story for each file in the directory'''
    train=[]
    test=[]
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        with open(path,'r') as g:
            try:
                data = json.load(g) 
                if 'test' in data['tags']:
                    #print(f'{path} is test')
                    new_KG = create_experiment_linearized(data)
                    #print(f'\n{new_KG=}')
                    test.append(new_KG)
                elif 'train' in data['tags']:
                    #print(f'{path} is train')
                    new_KG = create_experiment_linearized(data)
                    train.append(new_KG)

            except BaseException as e:
                print(f'The {path} file contains invalid JSON')
                

    return train, test


def main():
    """
    This function creates a json file that contains the linearized KG and the story
    """

    directory = 'DWIE/data/annos_with_content/'
<<<<<<< HEAD
    with open('DWIE_test.json', 'w') as f:
        f.write('[')
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
=======
    out_directory = 'Dataset/DWIE/'
>>>>>>> ba1db5e (Update preprocessing_DWIE.py)


    train_val, test = prepare_KG(directory,out_directory)

<<<<<<< HEAD
            with open(path) as g:
                try:
                    data = json.load(g) 
                    if 'test' in data['tags']:
                        print(path)
                        new_KG = create_experiment_linearized(data)
                        json.dump(new_KG, f, indent="") 
                        f.write(',\n')

                except BaseException as e:
                    print('The file contains invalid JSON')
                    print(path)
        f.write('{}')            
        f.write(']')
        f.close()

    with open('DWIE_train_val.json', 'w') as f:
        f.write('[')
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)



            with open(path) as g:
                try:
                    data = json.load(g) 
                    if 'train' in data['tags']:
                        new_KG = create_experiment_linearized(data)
                        json.dump(new_KG, f, indent="") 
                        f.write(',\n')

                except BaseException as e:
                    print('The file contains invalid JSON')
                    print(path)
        f.write('{}')            
        f.write(']')
        f.close()    



    with open('Dataset/DWIE/validation.json', 'w') as f:
        with open('DWIE_train_val.json') as g:
            data = json.load(g)
            data = data[:100]
            selected_data = []

            for d in data:
                if len(word_tokenize(d['story'])) < 1024:
                    selected_data.append(d)
            print(f'test len {len(data)} versus selected data len {len(selected_data)}')  
            
            json.dump(selected_data, f,indent="")  

    with open('Dataset/DWIE/train.json', 'w') as f:
        with open('DWIE_train_val.json') as g:
            data = json.load(g)
            data = data[100:-1]
            for d in data:
                if len(word_tokenize(d['story'])) < 1024:
                    selected_data.append(d)
            print(f'train len {len(data)} versus selected data len {len(selected_data)}')  
            json.dump(data, f,indent="")  

    with open('Dataset/DWIE/test.json', 'w') as f:
        with open('DWIE_test.json') as g:
            data = json.load(g)
            data = data[:-1]
            for d in data:
                if len(word_tokenize(d['story'])) < 1024:
                    selected_data.append(d)
            print(f'test len {len(data)} versus selected data len {len(selected_data)}') 
            json.dump(data, f,indent="")      
=======
    validation= train_val[:100]
    train = train_val[100:-1]

    print("Creating json for train...")
    with open(f'{out_directory}train.json', 'w') as f:
        train = remove_long_stories(train) 
        json.dump(train, f,indent=4)
    
    print("Creating json for validation...")
    with open(f'{out_directory}validation.json', 'w') as f:
        validation = remove_long_stories(validation)
        json.dump(validation, f,indent=4)
    
    print("Creating json for test...")
    with open(f'{out_directory}test.json', 'w') as f:
        test = remove_long_stories(test)
        json.dump(test, f,indent=4)
    
    print("DONE WITH CREATING THE GRAPHS")

   
>>>>>>> ba1db5e (Update preprocessing_DWIE.py)

def reification():
        check_gpu_availability()
        print("STARTING..")

        # SAVE THE DEVICE WE ARE WORKING WITH
        device = getting_device(gpu_prefence=True)
        for d in ["train","test","validation"]:

            print("Working on ", d)
            wrap(f"Dataset/DWIE/{d}.json")
            print("DONE with extracting reification for ", d)


if __name__ == "__main__":
    main()
    reification()
