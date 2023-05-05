import json
# Open the JSON file and load the data as a list of dictionaries


with open(f'Datasets/WebNLG/57core/57core_test.json', 'r') as f:
    test = json.load(f)


with open(f'results/webnlgResults/wenlg-Instances/stories.json', 'r+') as f:
    data = json.load(f)
    for d,t in zip(data,test):
        d['Instances_KG'] = test['Instances_KG']
    json.dump(data,f,indent=4,ensure_ascii = False)

        
with open(f'results/webnlgResults/wenlg-Types/stories.json', 'r+') as f:
    data = json.load(f)
    for d,t in zip(data,test):
        d['Types_KG'] = test['Types_KG']
    json.dump(data,f,indent=4,ensure_ascii = False)


with open(f'results/webnlgResults/wenlg-Subclasses/stories.json', 'r+') as f:
    data = json.load(f)
    for d,t in zip(data,test):   
        d['Subclasses_KG'] = test['Subclasses_KG']
    json.dump(data,f,indent=4,ensure_ascii = False)


 
