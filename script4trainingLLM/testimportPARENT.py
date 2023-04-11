import json
import os
import sys
from utils import *
#sys.path.append("parent/parent")

import evaluate



with open('Datasets/WebNLG/57_triples/oneClass/Trattini/test_57_oneClass.json','r') as f:
    data = json.load(f)

print(len(data))
data=data[2:5]
#print(data)
predictions=[]
reference=[]
graph=[]

for d in data:
    #print(d["graph"].split())
    
    predictions.append(d['story'])

    reference.append(d['story'])

    graph.append(d['Instances_KG'])





bleurt = evaluate.load("bleurt")
result_bleurt = bleurt.compute(predictions=predictions, references=reference)
print(f'{result_bleurt=}')




print(bleurt_metric(predictions,reference))