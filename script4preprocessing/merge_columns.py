
import json
# Open the JSON file and load the data as a list of dictionaries

with open('Datasets/WebNLG/57_triples/test_57.json', 'r') as f:
    data = json.load(f)

# Define the keys whose values should be merged
typeKG = ['Instances_KG', 'Types_KG']

subClassKG = ['Instances_KG', 'Types_KG','Subclasses_KG']

# Iterate over each dictionary in the list and merge the specified keys
for d in data:
    # Concatenate the values of the merge_keys into a single string
    merged_types = ' - '.join([d[k] for k in typeKG])
        
    merged_subClasse = ' - '.join([d[k] for k in subClassKG])

    
    # Choose a specific key to update with the merged value
    
    d['Types_KG'] = merged_types
    d['Subclasses_KG'] = merged_subClasse
    
    # Remove the values of the other merge keys
   
# Save the updated list of dictionaries back to the same JSON file
with open('/home/gabhoo/research/kg2Narrative/KGNarrative2/Datasets/WebNLG/57_triples/test.json', 'w') as f:
    f.write('[')
    for d in data:
        json.dump(d, f)
        f.write(',')
        f.write('\n')
    f.write('{}')    
    f.write(']')