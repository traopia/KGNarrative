# Knowledge Graph enhanced News Generation
The goal of this research project is to combine Large Language Models with Knowledge Graphs on the downstream task of Text generation, to research if infused knowledge enhance the quality and coherence of the generated text.


To run finetunemodel.py it is necessary to install bleurt from sourcecode as well as Parent from https://github.com/KaijuML/parent . Can also just comment out the scores. 

# DATASETS :
Two newly augmented are introduced based on the Existing WebNLG and DWIE. The enhanced version of these datasets can be found in the Datasets folder. 
The addition was done by either mining from text or scraping large knowledge bases. 
Recreating the augmentation can be done by running the scripts in Data_Preprocessing after the orginial dataset has been downloaded in the main folder. 
For each dataset the steps are:

## DWIE:
Downaloding (clones and dowloads the full dataset):
```
git clone https://github.com/klimzaporojets/DWIE
Python dwie_download.py
```
Preprocessing (GPU is necessary):
```
Python Data_Preprocessing preprocessing_DWIE.py

```


##WebNlg

Download WebNLG from orginial repo (https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0)
Release 3.0 in English is required
Preprocessing:
#### PROCESSING and DATA AUGMENTATION:
To preprocess the DWIE dataset (A GPU is necessary)
```
Python Data_Preprocessing/preprocessing_WebNLG.py
```





# MODELS
FOR THE FIRST FINETUNING ON EVENT NARRATIVE (parallel training not implemented yet)
python3 finetune_BART.py Datasets/EventNarrative EN Instance_Knowledge_Graph nonparallel megaBART2

FOR THE SECOND FINETUNING 


config deepspeed model parallelism : #https://github.com/pacman100/accelerate-deepspeed-test/blob/main/src/modeling/configs/zero2_config_accelerate.json




### Citations
Should you use this code/dataset for your own research, please cite: 
```

```


