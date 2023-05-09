#!/bin/bash

cols=('entities_list' 'semantic_of_news')

for element in "${cols[@]}"
do
    echo "FINETUNIING NOW ON $element"
    python3 KGNarrative/finetuning/finetunemodel_bart.py KGNarrative/Dataset/WebNLG $element bart-large KGNarrative/FINAL_RESULTS/WEBNLG/$element
    echo "DONE"
done


