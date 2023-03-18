import json


import json
from nltk.tokenize import word_tokenize



def select_short(file):
    with open(file, 'r') as f:
        data = json.load(f)
        

    selected_data = []

    for d in data:
        if len(word_tokenize(d['story'])) < 1024:
            selected_data.append(d)
    print(f'data len {len(data)} versus selected data len {len(selected_data)}')    
    return selected_data       


def save_json(data, file):
    with open(file, 'w',  encoding='utf-8') as f:
    # write the dictionary to the file as JSON
        json.dump(data, f, indent = 4)

def main():
    test = select_short('DWIE_semantics/test.json')
    save_json(test, 'DWIE_semantics/test_1024.json')  
    train = select_short('DWIE_semantics/train.json')
    save_json(train, 'DWIE_semantics/train_1024.json')
    val = select_short('DWIE_semantics/validation.json')
    save_json(val, 'DWIE_semantics/validation_1024.json')


if __name__ == '__main__':
    main()          
