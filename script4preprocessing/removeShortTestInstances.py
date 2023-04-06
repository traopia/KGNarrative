import json

with open("Datasets/WebNLG/57core/57core_test.json", "r") as f:
    test = json.load(f)

print(len(test),type(test))

for i in test:
    triple_string=i["Instances_KG"]
    if (triple_string.count('|')/2 -1) < 5:
        test.remove(i)  

print(len(test),type(test))

with open("Datasets/WebNLG/57core/57core_test_short.json", "w") as f:
    json.dump(test,f,indent=4,ensure_ascii = False)