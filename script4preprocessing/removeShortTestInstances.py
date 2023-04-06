import json

with open("Datasets/WebNLG/57core/57core_test_Alltriples.json", "r") as f:
    test = json.load(f)

print(len(test),type(test))
new_test=[]
for i in test:
    triple_string=i["Instances_KG"]
    if (triple_string.count('|')/2 -1) > 5:

        new_test.append(i)


print(len(test),type(test))

with open("Datasets/WebNLG/57core/57core_test_short.json", "w") as f:
    json.dump(new_test,f,indent=4,ensure_ascii = False)