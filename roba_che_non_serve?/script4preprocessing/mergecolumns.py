import json
# Open the JSON file and load the data as a list of dictionaries

def format_data(input_file, output_file):
    # splits=['train','val','test']
    # for split in splits:


        #with open(f'Datasets/withSummaryNotFinal/{split}_summary.json', 'r') as f:
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Define the keys whose values should be merged
        #print(data)

        instances = ['Instances_KG','core_descritpion']

        typeKG = ['Instances_KG', 'Types_KG','core_descritpion']

        subClassKG = ['Instances_KG', 'Types_KG','Subclasses_KG','core_descritpion']

        data=data[1:]

        # Iterate over each dictionary in the list and merge the specified keys
        for d in data:
            
            #print(d.keys()) 
            # Concatenate the values of the merge_keys into a single string

            d['core_descritpion'] = "Event - hasCore - "+ d['core_description']
            d.pop('core_description', None)

            #print(d.keys())

            merged_types = ' - '.join([d[k] for k in typeKG])

            merged_subClasse = ' - '.join([d[k] for k in subClassKG])

            merged_instances = ' - '.join([d[k] for k in instances])


            # Choose a specific key to update with the merged value

            d['Types_KG'] = merged_types
            d['Subclasses_KG'] = merged_subClasse
            d['Instances_KG'] = merged_instances

            # Remove the values of the other merge keys

        #with open('Datasets/WebNLG/57core/testtnocoma.json', 'w') as f:
        #    json.dump(data,f,indent=4,ensure_ascii = False)
        

        # Save the updated list of dictionaries back to the same JSON file
        #with open(f'/home/gabhoo/research/kg2Narrative/KGNarrative2/Datasets/WebNLG/57core/57core_{split}.json', 'w') as f:
        with open(output_file, 'w') as f:
            json.dump(data,f,indent=4,ensure_ascii = False)

# format_data('Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_dev.json','Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_dev.json')



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
        print(d[i])
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

format('Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_dev.json','Datasets/WebNLG/57_triples/Multiple_Classes/dev_57_MultipleClass.json','Datasets/WebNLG/57_triples/oneClass/Trattini/final/poppati_stupidi/oneClass_dev.json')


