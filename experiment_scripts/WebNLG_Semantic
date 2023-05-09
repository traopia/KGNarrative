#!/bin/bash

cols=('Instances_list' 'Instances_KG' 'Types_KG' 'Subclasses_KG'  'multi_Subclasses_KG')

for element in "${cols[@]}"
do
    echo "FINETUNIING NOW ON $element"
    python3 KGNarrative/finetuning/finetunemodel_bart.py KGNarrative/Dataset/WebNLG $element bart-large KGNarrative/FINAL_RESULTS/WEBNLG/$element
    echo "DONE"
done
