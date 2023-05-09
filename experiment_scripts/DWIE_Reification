#!/bin/bash

cols=('semantic_of_news' 'entities_list')

for element in "${cols[@]}"
do
        echo "FINETUNIING NOW ON $element"
        python3 KGNarrative/finetuning/finetunemodel_led.py KGNarrative/Dataset/DWIE $element led KGNarrative/FINAL_RESULTS/DWIE/$element
        echo "DONE"
done




