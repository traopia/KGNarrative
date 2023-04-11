import json
# Open the JSON file and load the data as a list of dictionaries

splits=['train','val','test']
for split in splits:


    with open(f'Datasets/withSummaryNotFinal/{split}_summary.json', 'r') as f:
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

        d['core_descritpion'] = "Event - hasCore - "+ d['core description']
        d.pop('core description', None)

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
    with open(f'/home/gabhoo/research/kg2Narrative/KGNarrative2/Datasets/WebNLG/57core/57core_{split}.json', 'w') as f:
                json.dump(data,f,indent=4,ensure_ascii = False)
