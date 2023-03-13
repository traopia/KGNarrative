# Knowledge Graph enhanced News Generation
The goal of this research project is to combine Large Language Models with Knowledge Graphs on the downstream task of Text generation, to research if infused knowledge enhance the quality and coherence of the generated text.


# DATASETS :

## Event Narrative:
To download the dataset [Event Narrative](https://www.kaggle.com/datasets/acolas1/eventnarration):
- download the folder and add it to the repository under the name EventNarrative


## DWIE:
To download the dataset [DWIE] (https://www.sciencedirect.com/science/article/pii/S0306457321000662):
```
python dwie_download.py
```

# PREPROCESSING:
To preprocess the DWIE dataset 
```
python DWIE_preprocessing.py
```
TODO: one py script with EventNarrative preprocessing 

# DATA AUGMENTATION
TOD0: make one py script with semantic augmentation pipeline


# MODELS
TODO: py script for all the finetunings 




### Citations
Should you use this code/dataset for your own research, please cite: 
```
@article{ZAPOROJETS2021102563,
title = {{DWIE}: An entity-centric dataset for multi-task document-level information extraction},
journal = {Information Processing & Management},
volume = {58},
number = {4},
pages = {102563},
year = {2021},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2021.102563},
url = {https://www.sciencedirect.com/science/article/pii/S0306457321000662},
author = {Klim Zaporojets and Johannes Deleu and Chris Develder and Thomas Demeester}
}
```


