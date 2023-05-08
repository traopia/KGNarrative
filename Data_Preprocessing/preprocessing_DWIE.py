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
            #print(data['concepts'][i]['concept'])
            
    for i,j in zip(range(len(data['relations'])),range(len(concept_text))):
        str += concept_text[data['relations'][i]['s']]+' - '+data['relations'][i]['p']+' - '+concept_text[data['relations'][i]['o']]+' | '
        #KG.append(concept_text[data['relations'][i]['s']]+' - '+data['relations'][i]['p']+' - '+concept_text[data['relations'][i]['o']] )

    return str.strip(' | ')



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
    f = open("DWIE/data/schema/ner.rdf", "r")
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
    d = {}
    #print('creating story')
    d['story'] = data['content'].replace('\n', ' ')
    #print(f'{d}')
    instances= create_linearized_KG(data)
    d['Instances_KG'] = instances
    types, concepts = create_types_KG(data)
    #print(instances)

    instance_list=instances.split('|')
    #print(instance_list)
    instance_list=[x.split(' - ')[a].strip() for x in instance_list if x != '' for a in [0,2] if x[a] != '']
    #instance_list=[x[a].strip() for x in d['Instances_KG'].split('') for a in [0,2] if x[a] != '']
    instance_list = " | ".join(list(set(instance_list)))

    d['Instances_list'] = instance_list
    #print(d['Instances_list'])

    #print("Created Instance_KG and TYPES ",d.keys())
   
    d['Types_KG'] = types + instances
    #print("Created Instance_KG and TYPES ",d.keys())
    subclasses = create_subclass_KG(data)
    #print(f'{subclasses=}')
    d['Subclasses_KG'] = subclasses + types + instances
    #print("Created Subclasses_KG ",d.keys())
    return d

def frequency_of_types(data):
    '''   This function returns a dictionary that contains the frequency of each type in the KG'''
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
    '''  This function returns a dictionary that contains the frequency of each type in the KG'''
    big_dict = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        with open(path) as g:
            try:
                data = json.load(g) 
                dicts = frequency_of_types(data)
                big_dict = Counter(big_dict) + Counter(dicts)

            except BaseException as e:
                print('Some files contain invalid JSON')
                # print('The file contains invalid JSON')
                # print(path)
    big_dict = sorted(big_dict.items(), key=lambda x:x[1], reverse=True)
    return big_dict

def prepare_KG(directory,outdir):
    '''  This function creates a linearized KG for each file in the directory and saves it in the outdir'''
    train=[]
    test=[]
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        with open(path,'r') as g:
            try:
                data = json.load(g) 
                if 'test' in data['tags']:
                    new_KG = create_experiment_linearized(data)
                    test.append(new_KG)
                elif 'train' in data['tags']:
                    #print(f'{path} is train')
                    new_KG = create_experiment_linearized(data)
                    train.append(new_KG)

            except BaseException as e:
                print(f'The {path} file contains invalid JSON')
                
    return train, test

def remove_long_stories(data):
    '''  This function removes the stories that are longer than 1024 tokens'''
    selected_data = []
    for d in data:
        if len(word_tokenize(d['story'])) < 1024:
            selected_data.append(d)       
    print(f'From len {len(data)} only {len(selected_data)} selected ')  
    return selected_data

def format():
    ''' This function formats the JSON file in the chosen format want'''

    for Dataset in ['test','train','validation']:
        with open(f"Dataset/DWIE/{Dataset}.json", 'r') as f:
            d = json.load(f)

            print(f'Loaded {Dataset} dataset. Formatting...')

            # Define the keys whose values should be merged
            instances = ['Instances_KG']

            typeKG = ['Instances_KG', 'Types_KG']

            subClassKG = ['Instances_KG', 'Types_KG','Subclasses_KG']
            #for d,d_s in zip(data,data_subclass):
            for i in range(len(d)):  
                #print(d[i])
                #print("[CORE] "+ d[i]['core_description'] +" [TRIPLES]")

                #MERGE CGRAPHS AND ADD CORE

                merged_types = "[CORE] "+ d[i]['core_description'] +" [TRIPLES] "+' | '.join([d[i][k] for k in typeKG])

                merged_subClasse = "[CORE] "+d[i]['core_description'] + " [TRIPLES] " + ' | '.join([d[i][k] for k in subClassKG])

                merged_instances = "[CORE] "+d[i]['core_description'] + " [TRIPLES] " + ' | '.join([d[i][k] for k in instances])


                d[i]['Types_KG'] = merged_types
                d[i]['Subclasses_KG'] = merged_subClasse
                d[i]['Instances_KG'] = merged_instances

                #ADD CORE TO ENTITIES LIST AND SEMANTIC OF NEWS

                #d[i]['Instances_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + " | ".join(d[i]['Instances_list'])
                #d[i]['entities_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + " | ".join(d[i]['entities_list'])

                
                d[i]['Instances_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + d[i]['Instances_list']
                d[i]['entities_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + d[i]['entities_list']


                d[i]['semantic_of_news'] = "[CORE] "+d[i]['core_description']+ " [TRIPLES] " + d[i]['semantic_of_news']
                with open(f"Dataset/DWIE/{Dataset}.json", 'w') as f:
                    json.dump(d,f,indent=4,ensure_ascii = False)    

def main():
    """
    This function creates a json file that contains the linearized KG and the story
    """

    directory = 'DWIE/data/annos_with_content/'
    out_directory = 'Dataset/DWIE/'


    train_val, test = prepare_KG(directory,out_directory)

    validation= train_val[:100]
    train = train_val[100:-1]

    print("Creating json for test...")
    with open(f'{out_directory}test.json', 'w') as f:
        test = remove_long_stories(test)
        json.dump(test, f,indent=4)
        
    print("Creating json for train...")
    with open(f'{out_directory}train.json', 'w') as f:
        train = remove_long_stories(train) 
        json.dump(train, f,indent=4)
    
    print("Creating json for validation...")
    with open(f'{out_directory}validation.json', 'w') as f:
        validation = remove_long_stories(validation)
        json.dump(validation, f,indent=4)
    

    
    print("DONE WITH CREATING THE GRAPHS")

  

def reification():
        check_gpu_availability()
        print("STARTING..")

        # SAVE THE DEVICE WE ARE WORKING WITH
        device = getting_device(gpu_prefence=True)
        for d in ["test","train","validation"]:

            print("Working on ", d)
            wrap(f"Dataset/DWIE/{d}.json")
            print("DONE with extracting reification for ", d)


if __name__ == "__main__":
    main()
    reification()
    format()
