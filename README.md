# Using Semantics for Content Planning: Lessons Learnt
This repo contains the code for "Using Semantics for Content Planning: Lessons Learnt", where different and deeper levels of semantic in a content planner are tested for textual generation. Two dataset are augmented with semantic information and tested on popular transformer models for language generation.

# DATASETS :
Two newly augmented are introduced based on the Existing WebNLG and DWIE. The enhanced version of these datasets can be found in the Datasets folder. 
The addition was done by either mining from text or scraping large knowledge bases. 
Recreating the augmentation can be done by running the scripts in Data_Preprocessing after the orginial dataset has been downloaded in the main folder. 
For each dataset the steps are:

## DWIE:
Downaloding (clones and dowloads the full dataset):
```
git clone https://github.com/klimzaporojets/DWIE
python dwie_download.py
```
Preprocessing (GPU is required):
```
python Data_Preprocessing/preprocessing_DWIE.py

```

## WebNlg

Download WebNLG from orginial repo (https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0)
Release 3.0 in English is required
Preprocessing (GPU is required):
```
python Data_Preprocessing/preprocessing_WebNLG.py
```

# MODELS
For the results, Bart-large was utilized with WebNLG and LongFormer (led) for DWIE.
For finetuning model on a specific content planner: ($element is one of 'Types_KG' 'Instances_KG' 'Subclasses_KG' 'Instances_list' 'multi_Subclasses_KG' 'entities_list' 'semantic_of_news')

```
#WebNLG
python3 script4trainingLLM/finetunemodel_webnlg.py Datasets/WebNLG/4experiment full $element bart-large path/to/results/$element --learning_rate 0.0001 --batch 1 --epochs 3
#DWIE
python3 script4trainingLLM/LED_$_DWIE.py Datasets/WebNLG/4experiment full $element bart-large path/to/results/$element --learning_rate 0.0001 --batch 1 --epochs 3
```

# RESULTS
To reproduce the results from the paper use the scripts in the scripts folder by running for example:
```
./scripts/webnlg_Semantic

```


### Citations
Should you use this code/dataset for your own research, please cite: 
```

```


