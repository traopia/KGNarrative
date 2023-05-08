# -*- coding: utf-8 -*-
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from SPARQLWrapper import SPARQLWrapper, JSON
import json
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
from tqdm import tqdm
import re
import shutil
#from script4preprocessing.mine_reificatedWEBNLG import *
from src_preprocessing.completeMiner import *

#MERGE FILES
def merge_xmls(input_file, output_file):
    """ 
    Merges all XML files in a directory into a single XML file.
    """
    # Path to the directory containing the XML files
    xml_dir = input_file

    # Initialize the root element for the merged XML file
    merged_root = ET.Element("root")

    # Loop through all XML files in the directory
    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            # Parse the XML file into an ElementTree object
            tree = ET.parse(os.path.join(xml_dir, filename))

            # Get the root element of the parsed XML
            root = tree.getroot()

            # Append the root element to the merged XML file
            merged_root.extend(root)

    # Create an ElementTree object for the merged XML
    merged_tree = ET.ElementTree(merged_root)

    # Write the merged XML to a file
    merged_tree.write(output_file)


def merge_of_merge():
    ''' 
    Merge the 5,6,7 triples files in the 57 triples files for both train and dev
    For test set we have to do it later'''

    if not os.path.exists('WebNLG/release_v3.0/en/selected'):
        os.makedirs('WebNLG/release_v3.0/en/selected')
    shutil.copy('WebNLG/release_v3.0/en/test/rdf-to-text-generation-test-data-with-refs-en.xml','WebNLG/release_v3.0/en/selected/test_triples.xml')

    for data in ['train','dev']:
        if not os.path.exists(f"WebNLG/release_v3.0/en/{data}/triples_57"):
            os.makedirs(f"WebNLG/release_v3.0/en/{data}/triples_57")
        for i in [5,6,7]:
            merge_xmls(f"WebNLG/release_v3.0/en/{data}/{i}triples", f"WebNLG/release_v3.0/en/{data}/triples_57/{data}_{i}triples.xml")
        merge_xmls(f"WebNLG/release_v3.0/en/{data}/triples_57", f"WebNLG/release_v3.0/en/selected/{data}_57triples.xml")


#CREATE JSON FILES in FORMAT

#get entity class
def get_entity_class(entity, subclass=False,multiple=False):
    '''construct the SPARQL query to retrieve the class of the entity
    :param entity: the entity to retrieve the class for
    :param subclass: if True, retrieve the subclass of the entity
    :param multiple: if True, retrieve all classes of the entity
    :return: the class of the entity'''

    print(f'Inside get_entity_class: {entity=}')
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?class ?label WHERE {
        <%s> rdf:type ?class .
        ?class rdfs:subClassOf* owl:Thing .
        ?class rdfs:label ?label .
        FILTER (lang(?label) = "en")
    }
    """ % entity 


    # set the SPARQL endpoint and query string
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # execute the query and retrieve the results
    results = sparql.query().convert()
    bindings = results['results']['bindings']
    if multiple:
        
        class_uri = []
        class_label = []
        for i in range(len(bindings)):
            class_uri.append(bindings[i]['class']['value'])
            class_label.append(bindings[i]['label']['value'])
        if class_uri != None:
            
            if subclass:
                return list(set(class_uri))[:3]
            else:
                return list(set(class_label))[:3]
    
    # extract the class label from the results
    if len(bindings) > 0 and bindings != None:

        class_uri = bindings[0]['class']['value']
        class_label = bindings[0]['label']['value']
        if class_uri != None:
            if subclass:
                return class_uri
            else:
                return class_label
#get entity superclass
def get_entity_subclass(entity):
    '''construct the SPARQL query to retrieve the class of the entity
    :param entity: the entity to retrieve the superclass for   
    :return: the class of the entity'''

    clas = entity
    print(f'Inside get_entity_subclass trying with {clas}')
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # construct the SPARQL query to retrieve the class of the entity
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?class ?label WHERE {
        <%s> rdfs:subClassOf ?class .

        ?class rdfs:label ?label .

    }
    """ % clas

    # set the SPARQL endpoint and query string
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # execute the query and retrieve the results
    results = sparql.query().convert()
    bindings = results['results']['bindings']

    # extract the class label from the results
    if len(bindings) > 0:
        class_uri = bindings[0]['class']['value']
        class_label = bindings[0]['label']['value']
        return class_label
    else:
        return None  

def create_dict_file(tree,verbose=False):
    '''Create a dictionary file from the XML file:
    :param tree: the ElementTree object representing the XML file
    :return: a list of dictionaries, where each dictionary represents an entry in the XML file designed according to oor format
    Story: the text of the story
    Instances_KG: the KG triples of the story
    Types_KG: the types of the entities in the story
    SubClasses_KG: the subclasses of the entities in the story
    multi_Types_KG: the types of the entities in the story (more than one class per entity)
    multi_SubClasses_KG: the subclasses of the entities in the story (more than one class per entity)
    '''

    # Initialize an empty list to store the results
    results = []

    # Iterate over all entry elements
    for entry in tree.findall('.//entry'):

        # Initialize an empty dictionary for this entry
        entry_dict = {}

        # Extract the eid attribute and store it as the entry ID
        #entry_dict['Entry ID'] = entry.get('eid')

        # Initialize an empty list to store the otriples for this entry
        otriples = []

        # Iterate over all otriple elements in this entry's originaltripleset
        for otriple in entry.findall('.//originaltripleset/otriple'):

            # Extract the text content of the otriple element and append it to the otriples list
            otriples.append(otriple.text.replace('|', '-'))

        # Join the otriples list into a single string, with each triple separated by '-'
        
        entry_dict['story'] = entry.find('.//lex').text
        entry_dict['Instances_KG'] = ' | '.join(otriples)
        #print(f'{entry_dict=}')

        #entities = [triple.split(' | ')[2] for triple in otriples]
        #entities.append(str(otriples[0].split(' | ')[0]))
        
        entities=[str(otriples[0].split(' - ')[0])]+[triple.split(' - ')[2] for triple in otriples]
        #print(type(entities),entities)
        entry_dict['Instances_list'] = ' | '.join(set(entities))
        entities = [entity.replace(" ", "_") for entity in entities]

        if verbose==True:
            print(f'{entry_dict=}')
        #classes
        if verbose==True:
            print('now querying classes:')
            print(f'{entities=}')
        classes = [get_entity_class('http://dbpedia.org/resource/'+entity) for entity in entities if '"' not in entity and '<' not in entity]
        classes = list(filter(lambda item: item is not None, classes))
        #classes uri
        if verbose==True:
            print('now querying classes uri:')
        classes_uri = [get_entity_class('http://dbpedia.org/resource/'+entity, subclass=True) for entity in entities if '"' not in entity and '<' not in entity]
        classes_uri = list(filter(lambda item: item is not None, classes_uri))
        
        if verbose==True:
            print(f'{classes=}')
            print(f'{classes_uri=}')

        entry_dict['Types_KG'] = ' | '.join(set([f"{entity.replace('_', ' ')} - type - {get_entity_class('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity and '<' not in entity  if get_entity_class('http://dbpedia.org/resource/'+entity) != None]))
        entry_dict['Subclasses_KG'] =  ' | '.join(set([f"{i} - subclass - {get_entity_subclass(j)}" for i,j in zip(classes, classes_uri)]))
        entry_dict['Instances_KG'] = entry_dict['Instances_KG'].replace('_', ' ')
        entry_dict['story'] = entry_dict['story'].replace('"',' ')

        if verbose==True:
            print(f'Inst-sub-types DONE:{entry_dict=}')

        # MULTI classes
        multi_classes = [get_entity_class('http://dbpedia.org/resource/'+entity,multiple=True)for entity in entities  if '"' not in entity and '<' not in entity]
        multi_classes = list(filter(lambda item: item is not None, classes))

        multi_classes_uri = [get_entity_class('http://dbpedia.org/resource/'+entity, subclass=True, multiple=True)for entity in entities  if '"' not in entity and '<' not in entity]
        multi_classes_uri = list(filter(lambda item: item is not None, classes_uri))

        entry_dict['multi_Types_KG'] = ' | '.join(set([f"{entity.replace('_', ' ')} - type - {get_entity_class('http://dbpedia.org/resource/'+entity, multiple=True)[i]}" for entity in entities if '"' not in entity and '<' not in entity  if get_entity_class('http://dbpedia.org/resource/'+entity) != None for i in range(len(get_entity_class('http://dbpedia.org/resource/'+entity, multiple=True)))]))
        entry_dict['multi_Subclasses_KG'] =  ' | '.join(set([f"{i} - subclass - {get_entity_subclass(j)}" for i,j in zip(multi_classes, multi_classes_uri) ]))

        if verbose==True:
            print(f'Multi Inst-sub-types and MULTI DONE:{entry_dict=}')
        dates = []
        s = entry_dict["Instances_KG"]
        date = re.findall(r'\d{4} - \d{2} - \d{2}', s)
        if date != []:
            dates.append(date)
            for j in date:
                entry_dict["Instances_KG"] = entry_dict["Instances_KG"].replace(j, j.replace(" - ", "/").replace('"'))

        """
        #change trattini format
        entry_dict['Instances_KG'] = entry_dict['Instances_KG'].replace(' - ','|').replace(' | ','-')
        entry_dict['Instances_KG'] = entry_dict['Instances_KG'].replace('-',' - ').replace('|',' | ')
        entry_dict['Types_KG'] = entry_dict['Types_KG'].replace(' - ','|').replace(' | ','-')
        entry_dict['Types_KG'] = entry_dict['Types_KG'].replace('-',' - ').replace('|',' | ')
        entry_dict['Subclasses_KG'] = entry_dict['Subclasses_KG'].replace(' - ','|').replace(' | ','-')
        entry_dict['Subclasses_KG'] = entry_dict['Subclasses_KG'].replace('-',' - ').replace('|',' | ')
        entry_dict['multi_Types_KG'] = entry_dict['multi_Types_KG'].replace(' - ','|').replace(' | ','-')
        entry_dict['multi_Types_KG'] = entry_dict['multi_Types_KG'].replace('-',' - ').replace('|',' | ')
        entry_dict['multi_Subclasses_KG'] = entry_dict['multi_Subclasses_KG'].replace(' - ','|').replace(' | ','-')
        entry_dict['multi_Subclasses_KG'] = entry_dict['multi_Subclasses_KG'].replace('-',' - ').replace('|',' | ')
        """

        entry_dict['story'] = entry_dict['story'].replace('"',' ')

        #print(entry_dict)     

        # Append the entry dictionary to the results list   
        results.append(entry_dict)
    return results


def create_file_format():
    ''' 
    This function creates a JSON file from the WebNLG Dataset,
    It does the preprocessing of the data and it creates the file in the format we want
    '''

    
    #load the WebNLG Dataset from the XML file

    print("Creating Train File\n")
    tree = ET.parse(f"WebNLG/release_v3.0/en/selected/train_57triples.xml")
    root = tree.getroot()
    data = create_dict_file(tree,verbose=True)
    print("Train data:",data)
    with open(f"Dataset/WebNLG/train.json", 'w') as f:
        json.dump(data, f, indent = 4)
    print("Train File Created\n\n")

    """
    print("Creating Test File\n")
    tree = ET.parse(f"WebNLG/release_v3.0/en/selected/test_triples.xml")
    root = tree.getroot()
    data = create_dict_file(tree)
    print("Test data:",data)
    with open(f"Dataset/WebNLG/test.json", 'w') as f:
        json.dump(data, f, indent = 4)
    print("Test File Created\n\n")


    print("Creating Validation File\n")
    tree = ET.parse(f"WebNLG/release_v3.0/en/selected/dev_57triples.xml")
    root = tree.getroot()
    data = create_dict_file(tree)
    print("Validation data:",data)
    with open(f"Dataset/WebNLG/validation.json", 'w') as f:
        json.dump(data, f, indent = 4)
    print("Validation File Created\n\n")
    """
    





   
  



##ADD SPECIFIC STUFF TO THE JSON FILE

def remove_short_test():

    with open("Dataset/WebNLG/test.json", "r") as f:
        test = json.load(f)

    print(len(test),type(test))
    new_test=[]
    for i in test:
        triple_string=i["Instances_KG"]
        if (triple_string.count('-')/2 -1) > 5:

            new_test.append(i)


    print(len(test),type(test))

    with open("Dataset/WebNLG/test.json", "w") as f:
        json.dump(new_test,f,indent=4,ensure_ascii = False)







def remove_if_not_reification():
    ''' This function pops the observation for which the semantic of the news pipeline didnt work'''

    for d in ['validation', 'train', 'test']:
        with open(f"Dataset/WebNLG/{d}.json", "r") as f:
            data = json.load(f)
        result = [data[i] for i in range(len(data)) if ' | aoh' not in (data[i]["semantic_of_news"]+'aoh')]
        index_pop = [i for i in range(len(data)) if ' | aoh' in (data[i]["semantic_of_news"]+'aoh')]
        print('Number of data points for which reification extraction didnt work: ',len(index_pop))
        result = [data[i] for i in range(len(data)) if i not in index_pop]
        with open(f"Dataset/WebNLG/{d}.json", 'w') as f:
            json.dump(result, f, indent = 4) 

        return index_pop

                         



def format():
    ''' This function formats the JSON file in the chosen format want'''

    for Dataset in ['test','training','validation']:
        with open(f"Dataset/WebNLG/{Dataset}.json", 'w') as f:
            d = json.load(f)

 
    
            # Define the keys whose values should be merged
            instances = ['Instances_KG']

            typeKG = ['Instances_KG', 'Types_KG']

            subClassKG = ['Instances_KG', 'Types_KG','Subclasses_KG']
            #for d,d_s in zip(data,data_subclass):
            for i in range(len(d)):  
                #print(d[i])
                print("[CORE] "+ d[i]['core_description'] +" [TRIPLES]")

                #MERGE CGRAPHS AND ADD CORE

                merged_types = "[CORE] "+ d[i]['core_description'] +" [TRIPLES] "+' | '.join([d[i][k] for k in typeKG])

                merged_subClasse = "[CORE] "+d[i]['core_description'] + " [TRIPLES] " + ' | '.join([d[i][k] for k in subClassKG])

                merged_instances = "[CORE] "+d[i]['core_description'] + " [TRIPLES] " + ' | '.join([d[i][k] for k in instances])

                d[i]['multi_Subclasses_KG'] = "[CORE] "+ d[i]['core_description'] +" [TRIPLES] "+' | '.join([d[i][k] for k in typeKG]) + d[i]['multi_Subclasses_KG']
                d[i]['multi_Types_KG'] = "[CORE] "+ d[i]['core_description'] +" [TRIPLES] "+' | '.join([d[i][k] for k in instances]) + d[i]['multi_Types_KG']

                d[i]['Types_KG'] = merged_types
                d[i]['Subclasses_KG'] = merged_subClasse
                d[i]['Instances_KG'] = merged_instances

                #ADD CORE TO ENTITIES LIST AND SEMANTIC OF NEWS

                #d[i]['Instances_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + " | ".join(d[i]['Instances_list'])
                #d[i]['entities_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + " | ".join(d[i]['entities_list'])


                d[i]['Instances_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + d[i]['Instances_list']
                d[i]['entities_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + d[i]['entities_list']


                d[i]['semantic_of_news'] = "[CORE] "+d[i]['core_description']+ " [TRIPLES] " + d[i]['semantic_of_news']
                with open(f"Dataset/WebNLG/{Dataset}.json", 'w') as f:
                    json.dump(d,f,indent=4,ensure_ascii = False)        



    
    



def reification():
        check_gpu_availability()
        print("STARTING..")

        # SAVE THE DEVICE WE ARE WORKING WITH
        device = getting_device(gpu_prefence=True)
        for d in ["train","test","validation"]:

            print("Working on ", d)
            wrap(f"Dataset/WebNLG/{d}.json")
            print("DONE with extracting reification for ", d)
