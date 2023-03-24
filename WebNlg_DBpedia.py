from SPARQLWrapper import SPARQLWrapper, JSON
import os
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from SPARQLWrapper import SPARQLWrapper, JSON
import json
# set up the SPARQL endpoint for DBpedia
sparql = SPARQLWrapper("http://dbpedia.org/sparql")



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






def get_entity_class(entity):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # construct the SPARQL query to retrieve the class of the entity
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

    # extract the class label from the results
    if len(bindings) > 0:
        class_uri = bindings[0]['class']['value']
        class_label = bindings[0]['label']['value']
        return class_label
    else:
        return None


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
        entry_dict['Instances KG'] = ' - '.join(otriples)
        entities = [triple.split(' | ')[2] for triple in otriples]
        entities.append(str(otriples[0].split(' | ')[0]))
        print(entities)
        entities = [entity.replace(" ", "_") for entity in entities]
        # a = [get_entity_class('http://dbpedia.org/resource/'+entity) for entity in entities if '"' not in entity]
        entry_dict['Types KG'] =  ' - '.join([f"{entity.replace(' ', '_')} | type | {get_entity_class('http://dbpedia.org/resource/'+entity)}" for entity in entities if '"' not in entity])

        print(entry_dict['Types KG'])
        

        # Append the entry dictionary to the results list
        results.append(entry_dict)
        return results



def to_json(results, output_json):
    #with open('Datasets/WebNLG/dev_7triples.json', 'w') as f:
    with open(output_json, 'w') as f:
        f.write('[')
        for i in results:
            json.dump(i, f)
            f.write(',')
            f.write('\n')
        f.write('{}')    
        f.write(']')

def main(file_to_preprocess, output_file):
    # set up the SPARQL endpoint for DBpedia
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # load the WebNLG dataset from the XML file
    #tree = ET.parse("WebNLG/release_v3.0/en/dev/7triples/dev_7triples.xml")
    #tree = ET.parse("WebNLG/release_v3.0/en/test/rdf-to-text-generation-test-data-with-refs-en.xml")

    tree = ET.parse(file_to_preprocess)
    root = tree.getroot()
    results = create_dict_file(tree)
    to_json(results, output_file)


if __name__ == "__main__":
    main("WebNLG/release_v3.0/en/test/rdf-to-text-generation-test-data-with-refs-en.xml", "Datasets/WebNLG/test_1.json")


