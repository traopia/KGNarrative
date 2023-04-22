from SPARQLWrapper import SPARQLWrapper, JSON
import os
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from SPARQLWrapper import SPARQLWrapper, JSON
import json
# set up the SPARQL endpoint for DBpedia
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
from tqdm import tqdm



def merge_xmls(input_file, output_file):
    """ 
    Merges all XML files in a directory into a single XML file.
    """
    # Path to the directory containing the XML files
    #xml_dir = "WebNLG/release_v3.0/en/dev/7triples/"
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
    #merged_tree.write("WebNLG/release_v3.0/en/dev/7triples/dev_7triples.xml")
    merged_tree.write(output_file)






# def get_entity_class(entity, subclass=False):
#     # construct the SPARQL query to retrieve the class of the entity
#     query = """
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     SELECT ?class ?label WHERE {
#         <%s> rdf:type ?class .
#         ?class rdfs:subClassOf* owl:Thing .
#         ?class rdfs:label ?label .
#         FILTER (lang(?label) = "en")
#     }
#     """ % entity

#     # set the SPARQL endpoint and query string
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)

#     # execute the query and retrieve the results
#     results = sparql.query().convert()
#     bindings = results['results']['bindings']

#     # extract the class label from the results
#     if len(bindings) > 0:
#         class_uri = bindings[0]['class']['value']
#         class_label = bindings[0]['label']['value']
#         if subclass:
#             return class_uri
#         else:
#             return class_label
#     else:
#         return None

# def get_entity_subclass(entity):
#     clas = get_entity_class(entity, subclass = True)
#     print(clas)
#     sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#     # construct the SPARQL query to retrieve the class of the entity
#     query = """
#     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     SELECT ?class ?label WHERE {
#         <%s> rdfs:subClassOf ?class .
#         ?class rdfs:subClassOf* owl:Thing .
#         ?class rdfs:label ?label .
#         FILTER (lang(?label) = "en")
#     }
#     """ % clas

#     # set the SPARQL endpoint and query string
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)

#     # execute the query and retrieve the results
#     results = sparql.query().convert()
#     bindings = results['results']['bindings']

#     # extract the class label from the results
#     if len(bindings) > 0:
#         class_uri = bindings[0]['class']['value']
#         class_label = bindings[0]['label']['value']
#         return class_label
#     else:
#         return None    


def create_dict_file(tree):

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
            otriples.append(otriple.text)
        print(otriples)

        # Join the otriples list into a single string, with each triple separated by '-'
        
        entry_dict['story'] = entry.find('.//lex').text
        entry_dict['Instances_KG'] = ' - '.join(otriples)
        entities = [triple.split(' | ')[2] for triple in otriples]
        entities.append(str(otriples[0].split(' | ')[0]))
        print(entities)
        entities = [entity.replace(" ", "_") for entity in entities]
        # a = [get_entity_class('http://dbpedia.org/resource/'+entity) for entity in entities if '"' not in entity]
        # entry_dict['Types KG'] =  ' - '.join([f"{entity.replace(' ', '_')} | type | {get_entity_class('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity])
        # entry_dict['Subclasses KG'] =  ' - '.join([f"{entity.replace(' ', '_')} | subclass | {get_entity_subclass('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity])

        entry_dict['Types_KG'] =  ' - '.join(set([f"{entity.replace(' ', '_')} | type | {get_entity_class('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity]))
        print(entry_dict['Types_KG'])
        #entry_dict['Subclasses KG'] =  ' - '.join([f"{entity.replace(' ', '_')} | subclass | {get_entity_subclass('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity])
        entry_dict['Subclasses_KG'] =  ' - '.join(set([f"{get_entity_class('http://dbpedia.org/resource/'+entity)} | subclass | {get_entity_subclass('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity]))
        entry_dict['Subclasses_KG'] =  entry_dict['Types_KG'] + ' - ' + entry_dict['Subclasses_KG']
        #print(entry_dict['Types KG'])
        print(entry_dict['Subclasses_KG'])
        

        # Append the entry dictionary to the results list
        results.append(entry_dict)
    return results



# def to_json(results, output_json):
#     #with open('Datasets/WebNLG/dev_7triples.json', 'w') as f:
#     with open(output_json, 'w') as f:
#         f.write('[')
#         for i in results:
#             json.dump(i, f)
#             f.write(',')
#             f.write('\n')
#         f.write('{}')    
#         f.write(']')


def get_entity_class(entity, subclass=False,multiple=False):
    #construct the SPARQL query to retrieve the class of the entity


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

    

def get_entity_subclass(entity):
    clas = entity
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

def get_onec_class(data, output_file):
    for i in tqdm(range(len(data))):
        #entities
        entities = [triple.split(' | ')[2] for triple in data[i]['Instances_KG'].split(' - ')]
        entities.append(str(data[i]['Instances_KG'].split(' | ')[0]))
        entities = [entity.replace(' ', '_') for entity in entities]
        #print(entities)
        #print([('http://dbpedia.org/resource/'+entity) for entity in entities])

        #classes
        classes = [get_entity_class('http://dbpedia.org/resource/'+entity)for entity in entities if '"' not in entity ]
        classes = list(filter(lambda item: item is not None, classes))
        #classes uri
        classes_uri = [get_entity_class('http://dbpedia.org/resource/'+entity, subclass=True)for entity in entities if '"' not in entity ]
        classes_uri = list(filter(lambda item: item is not None, classes_uri))


        #KG
        data[i]['Types_KG'] = ' - '.join(set([f"{entity.replace('_', ' ')} | type | {get_entity_class('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity if get_entity_class('http://dbpedia.org/resource/'+entity) != None]))
        #data[i]['Types_KG'] = ' - '.join(set([f"{entity} | type | {c}" for entity,c in zip(entities,classes) if '"' not in entity ]))
        print(data[i]['Types_KG'])
        data[i]['Subclasses_KG'] =  ' - '.join(set([f"{i} | subclass | {get_entity_subclass(j)}" for i,j in zip(classes, classes_uri)]))
        print(data[i]['Subclasses_KG'])
        data[i]['Instances_KG'] = data[i]['Instances_KG'].replace('_', ' ')
        print(data[i]['Instances_KG'])
        data[i]['story'] = data[i]['story'].replace('"',' ')


    with open(output_file, 'w') as f:
        json.dump(data, f, indent = 4)



def get_multiple_class(data, output_file):
    for i in tqdm(range(len(data))):
        #entities
        entities = [triple.split(' - ')[2] for triple in data[i]['Instances_KG'].split(' | ')]
        entities.append(str(data[i]['Instances_KG'].split(' | ')[0]))
        entities = [entity.replace(' ', '_') for entity in entities]
        #print(entities)
        #print([('http://dbpedia.org/resource/'+entity) for entity in entities])

        #classes
        classes = [get_entity_class('http://dbpedia.org/resource/'+entity,multiple=True)for entity in entities if '"' not in entity ]
        classes = list(filter(lambda item: item is not None, classes))
        #print(classes)
        #classes uri
        classes_uri = [get_entity_class('http://dbpedia.org/resource/'+entity, subclass=True, multiple=True)for entity in entities if '"' not in entity ]
        classes_uri = list(filter(lambda item: item is not None, classes_uri))

        # print(classes_uri)
        # print(classes)


        #KG
        data[i]['Types_KG'] = ' - '.join(set([f"{entity.replace('_', ' ')} | type | {get_entity_class('http://dbpedia.org/resource/'+entity, multiple=True)[i]}" for entity in entities if '"' not in entity if get_entity_class('http://dbpedia.org/resource/'+entity) != None for i in range(len(get_entity_class('http://dbpedia.org/resource/'+entity, multiple=True)))]))
        #data[i]['Types_KG'] = ' - '.join(set([f"{entity} | type | {c}" for entity,c in zip(entities,classes) if '"' not in entity ]))
        print(data[i]['Types_KG'])
        
        data[i]['Subclasses_KG'] =  ' - '.join(set([f"{s} | subclass | {get_entity_subclass(t)}" for i,j in zip(classes, classes_uri) for s,t in zip(i,j)]))
        print(data[i]['Subclasses_KG'])
        data[i]['Instances_KG'] = data[i]['Instances_KG'].replace('_', ' ')
        data[i]['story'] = data[i]['story'].replace('"',' ')



    with open(output_file, 'w') as f:
        json.dump(data, f, indent = 4)


def trattini(file_to_preprocess, output_file):
    with open(file_to_preprocess, "r") as jsonFile:
        data = json.load(jsonFile)
    for i in range(len(data)):
        data[i]['Instances_KG'] = data[i]['Instances_KG'].replace(' - ','|').replace(' | ','-')
        data[i]['Instances_KG'] = data[i]['Instances_KG'].replace('-',' - ').replace('|',' | ')
        data[i]['Types_KG'] = data[i]['Types_KG'].replace(' - ','|').replace(' | ','-')
        data[i]['Types_KG'] = data[i]['Types_KG'].replace('-',' - ').replace('|',' | ')
        data[i]['Subclasses_KG'] = data[i]['Subclasses_KG'].replace(' - ','|').replace(' | ','-')
        data[i]['Subclasses_KG'] = data[i]['Subclasses_KG'].replace('-',' - ').replace('|',' | ')
        data[i]['story'] = data[i]['story'].replace('"',' ')

    import re

    dates = []
    for i in range(len(data)):
        s = data[i]["Instances_KG"]
        date = re.findall(r'\d{4} - \d{2} - \d{2}', s)
        if date != []:
            dates.append(date)
            for j in date:
                data[i]["Instances_KG"] = data[i]["Instances_KG"].replace(j, j.replace(" - ", "/").replace('"'))
                print(data[i]["Instances_KG"])    
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent = 4)

def instance_list(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    for i in range(len(data)):
        #input_string = data[i]['Instances_KG']
        input_string = data[i]['Instances']
        triples = input_string.split(' | ')

        # Loop through each triple and extract the first and third elements using the '-' delimiter
        list_instances = []
        for triple in triples:
            elements = triple.split(' - ')
            list_instances.append(elements[0])
            list_instances.append(elements[2])
        data[i]['Instances_list']  = list(set(list_instances))
    with open(input_file, 'w') as f:
            json.dump(data, f, indent = 4)         

def in_ist_DWIE(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
            #input_string = data[i]['Instances_KG']
            input_string = data[i]['Instances']
            triples = input_string.split(' | ')

            # Loop through each triple and extract the first and third elements using the '-' delimiter
            list_instances = []
            for triple in triples:
                elements = triple.split(' - ')
                if len(elements) == 3:
                    list_instances.append(elements[0])
                    list_instances.append(elements[2])
            data[i]['Instances_list']  = list(set(list_instances))
            print(data[i]['Instances_list'])
        
    with open(input_file, 'w') as f:
        json.dump(data, f, indent = 4)                    


def pop_stupid_boys(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    result = [data[i] for i in range(len(data)) if ' | aoh' not in (data[i]["semantic_of_news"]+'aoh')]
    index_pop = [i for i in range(len(data)) if ' | aoh' in (data[i]["semantic_of_news"]+'aoh')]
    #print(index_pop)
    print(len(index_pop))
    with open(output_file, 'w') as f:
        json.dump(result, f, indent = 4) 

    return index_pop

def pop_stupid_boys_multiple(input_file, output_file, index_pop):
    with open(input_file, "r") as f:
        data = json.load(f)
    result = [data[i] for i in range(len(data)) if i not in index_pop]
    with open(output_file, 'w') as f:
        json.dump(result, f, indent = 4)    

def format(input_file,input_file_subclasses, output_file):
    with open(input_file, 'r') as f:
        d = json.load(f)
    with open(input_file_subclasses, 'r') as f:
            d_s = json.load(f)    
    
    # Define the keys whose values should be merged
    instances = ['Instances_KG']

    typeKG = ['Instances_KG', 'Types_KG']

    subClassKG = ['Instances_KG', 'Types_KG','Subclasses_KG']
    print(d)
    #for d,d_s in zip(data,data_subclass):
    for i in range(len(d)):  
        #print(d[i])
        print("[CORE] "+ d[i]['core_description'] +" [TRIPLES]")

        #MERGE CGRAPHS AND ADD CORE

        merged_types = "[CORE] "+ d[i]['core_description'] +" [TRIPLES] "+' | '.join([d[i][k] for k in typeKG])

        merged_subClasse = "[CORE] "+d[i]['core_description'] + " [TRIPLES] " + ' | '.join([d[i][k] for k in subClassKG])

        merged_instances = "[CORE] "+d[i]['core_description'] + " [TRIPLES] " + ' | '.join([d[i][k] for k in instances])

        d[i]['multi_Subclasses_KG'] = "[CORE] "+ d[i]['core_description'] +" [TRIPLES] "+' | '.join([d[i][k] for k in typeKG]) + d_s[i]['Subclasses_KG']


        d[i]['Types_KG'] = merged_types
        d[i]['Subclasses_KG'] = merged_subClasse
        d[i]['Instances_KG'] = merged_instances

        #ADD CORE TO ENTITIES LIST AND SEMANTIC OF NEWS

        d[i]['Instances_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + " | ".join(d[i]['Instances_list'])
        d[i]['entities_list'] = "[CORE] "+d[i]['core_description']+ " [ENTITIES] " + " | ".join(d[i]['entities_list'])

        d[i]['semantic_of_news'] = "[CORE] "+d[i]['core_description']+ " [TRIPLES] " + d[i]['semantic_of_news']
    with open(output_file, 'w') as f:
        json.dump(d,f,indent=4,ensure_ascii = False)        


def main(file_to_preprocess, output_file):
    # set up the SPARQL endpoint for DBpedia
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # load the WebNLG dataset from the XML file
    #tree = ET.parse("WebNLG/release_v3.0/en/dev/7triples/dev_7triples.xml")
    #tree = ET.parse("WebNLG/release_v3.0/en/test/rdf-to-text-generation-test-data-with-refs-en.xml")

    # tree = ET.parse(file_to_preprocess)
    # root = tree.getroot()
    #  data = create_dict_file(tree)

    # with open(output_file, 'w') as f:
    #     json.dump(data, f, indent = 4)

    #get one class
    # with open(file_to_preprocess, "r") as jsonFile:
    #     data = json.load(jsonFile)
    # # get_onec_class(data, output_file)
    # #get multiple class
    # get_multiple_class(data, output_file)

    #trattini(file_to_preprocess, output_file)








if __name__ == "__main__":
    # merge_xmls("WebNLG/release_v3.0/en/dev/triples_dev_57", "WebNLG/release_v3.0/en/dev/dev_57triples.xml")
    # merge_xmls("WebNLG/release_v3.0/en/train/triples_train_57", "WebNLG/release_v3.0/en/train/train_57triples.xml")

    #main("WebNLG/release_v3.0/en/test/rdf-to-text-generation-test-data-with-refs-en.xml", "Datasets/WebNLG/test_1.json")
    #main("WebNLG/release_v3.0/en/train/train_57triples.xml", "Datasets/WebNLG/train_57_1.json")


    #main("Datasets/WebNLG/57_triples/dev_57.json", "Datasets/WebNLG/57_triples/dev_57_oneClass.json")
    #main("Datasets/WebNLG/57_triples/oneClass/test_57_oneClass.json","Datasets/WebNLG/57_triples/oneClass/test_57_oneClass.json")
    #main("Datasets/WebNLG/57_triples/Multiple_Classes/dev_57_MultipleClass.json","Datasets/WebNLG/57_triples/Multiple_Classes/dev_57_MultipleClass.json")
    #instance_list("Datasets/WebNLG/57_triples/oneClass/Trattini/train_57_oneClass.json")



    #index_pop = pop_stupid_boys("Datasets/WebNLG/57_triples/oneClass/Trattini/final/oneClass_dev.json", "Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_dev.json")
    # pop_stupid_boys_multiple('Datasets/WebNLG/57_triples/Multiple_Classes/dev_57_MultipleClass.json', 'Datasets/WebNLG/57_triples/Multiple_Classes/poppati_stupidi/dev_57_MultipleClass.json', index_pop)
    # index_pop = pop_stupid_boys("Datasets/WebNLG/57_triples/oneClass/Trattini/final/oneClass_test.json", "Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_test.json")
    # pop_stupid_boys_multiple('Datasets/WebNLG/57_triples/Multiple_Classes/test_57_MultipleClass.json', 'Datasets/WebNLG/57_triples/Multiple_Classes/poppati_stupidi/test_57_MultipleClass.json', index_pop)
    # index_pop = pop_stupid_boys("Datasets/WebNLG/57_triples/oneClass/Trattini/final/oneClass_train.json", "Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_train.json")
    # pop_stupid_boys_multiple('Datasets/WebNLG/57_triples/Multiple_Classes/train_57_MultipleClass.json', 'Datasets/WebNLG/57_triples/Multiple_Classes/poppati_stupidi/train_57_MultipleClass.json', index_pop)

    #splits =['train','dev','test']
    splits = ['train','validation','test']

    for d in splits:
        #print(d , pop_stupid_boys(f"Datasets/WebNLG/57_triples/oneClass/Trattini/final/oneClass_{d}.json", f"Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_{d}.json"))

        #format(f'Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_{d}.json',f'Datasets/WebNLG/57_triples/Multiple_Classes/{d}_57_MultipleClass.json',f'Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_{d}.json')
        in_ist_DWIE(f'Datasets/DWIE/DWIE_cleaned/{d}_cleaned_mined.json')
