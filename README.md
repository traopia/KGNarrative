# Knowledge Graph enhanced News Generation
The goal of this research project is to combine Large Language Models with Knowledge Graphs on the downstream task of Text generation, to research if infused knowledge enhance the quality and coherence of the generated text.


To run finetunemodel.py it is necessary to install bleurt from sourcecode as well as Parent from https://github.com/KaijuML/parent . Can also just comment out the scores. 

# DATASETS :
Two datasets are used in this project and can be found in /Dataset folder. 

## Event Narrative:
To download the original dataset  [Event Narrative](https://www.kaggle.com/datasets/acolas1/eventnarration):

The dataset has been preprocessed and the version in use can be downloaded from https://drive.google.com/drive/folders/1YuwHx6qRaOuf1Cfr87P2_445BdFMNqzr?usp=share_link
- download and extract the train test validation in the EventNarrative folder


## DWIE:
To download the dataset [DWIE](https://www.sciencedirect.com/science/article/pii/S0306457321000662):
```
python dwie_download.py
```

#### PREPROCESSING:
To preprocess the DWIE dataset 
```
python DWIE_preprocessing.py
```
TODO: one py script with EventNarrative preprocessing 

#### DATA AUGMENTATION
TOD0: make one py script with semantic augmentation pipeline


# MODELS
FOR THE FIRST FINETUNING ON EVENT NARRATIVE (parallel training not implemented yet)
python3 finetune_BART.py Datasets/EventNarrative EN Instance_Knowledge_Graph nonparallel megaBART2

FOR THE SECOND FINETUNING 


config deepspeed model parallelism : #https://github.com/pacman100/accelerate-deepspeed-test/blob/main/src/modeling/configs/zero2_config_accelerate.json




### Citations
Should you use this code/dataset for your own research, please cite: 
```

```


