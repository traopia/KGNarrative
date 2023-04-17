import json
import os
import sys
from utils import *
#sys.path.append("parent/parent")

import evaluate
import numpy as np



with open('results/webnlgResults/webnlg_semantic_results/Instances_KG/stories_withInput.json','r') as f:
    data = json.load(f)

#print(len(data))
data=data[2:3]
#print(data)
predictions=[]
reference=[]
graph=[]
print("\n\n")
for d in data:

    print(d['graph'])

    generated_story=d['generated_story']    
    print(f"{generated_story=}")
    predictions.append(generated_story)

    target = d['target']
    print(f"{target=}")
    reference.append(target)
    print('\n')
    #graph.append(d['Instances_KG'])



#predictions=['The cat was hunting the bird on the tree', 'A cat is trying to eat a bird on a tree']
#reference=['A cat is hunting a bird on a tree', 'A cat is trying to eat a bird on a tree']
#predictions = ["Apple cake", "horse camel"]
#references = ["hello there", "general kenobi"]
bleurt = evaluate.load("bleurt",'bleurt-large-512',module_type="metric")
#print(predictions,reference)
result_bleurt = bleurt.compute(predictions=predictions, references=reference)
#result_bleurt['scores']=np.mean(result_bleurt['scores'])
print(f'{result_bleurt=}')

#graph=[g.split('[TRIPLES]')[1] for g in graph]
#print(graph)

#print(parent_metric(predictions,reference,graph))
