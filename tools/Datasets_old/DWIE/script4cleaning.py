'''
gpe0, gpe1, gpe2, gpe0-x, gpe1-x, gpe2-x => geopolitical organization

in0 => in

loc => location

ngo => non-governmental organization

igo => intergovernmental organization

per => person

gov_org	=> governmental organizzation

org => organization

misc => miscellaneous entities

'''

import json

# FOR ORGANIZATION

def replace_gpe(data):
    if isinstance(data, str):
        return data.replace('gpe', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_gpe(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_gpe(value) for key, value in data.items()}
    else:
        return data

def replace_gpe0(data):
    if isinstance(data, str):
        return data.replace('gpe0', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_gpe0(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_gpe0(value) for key, value in data.items()}
    else:
        return data

def replace_gpe1(data):
    if isinstance(data, str):
        return data.replace('gpe1', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_gpe1(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_gpe1(value) for key, value in data.items()}
    else:
        return data

def replace_gpe2(data):
    if isinstance(data, str):
        return data.replace('gpe2', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_gpe2(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_gpe2(value) for key, value in data.items()}
    else:
        return data

def replace_gpe0x(data):
    if isinstance(data, str):
        return data.replace('gpe0-x', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_gpe0x(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_gpe0x(value) for key, value in data.items()}
    else:
        return data

def replace_gpe1x(data):
    if isinstance(data, str):
        return data.replace('gpe1-x', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_gpe1x(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_gpe1x(value) for key, value in data.items()}
    else:
        return data

def replace_gpe2x(data):
    if isinstance(data, str):
        return data.replace('gpe2-x', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_gpe2x(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_gpe2x(value) for key, value in data.items()}
    else:
        return data

# FOR IN0

def replace_in0(data):
    if isinstance(data, str):
        return data.replace('in0', 'geopolitical organization')
    elif isinstance(data, list):
        return [replace_in0(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_in0(value) for key, value in data.items()}
    else:
        return data

# FOR LOC

def replace_loc(data):
    if isinstance(data, str):
        return data.replace(' loc ', ' location ')
    elif isinstance(data, list):
        return [replace_loc(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_loc(value) for key, value in data.items()}
    else:
        return data

# FOR NGO

def replace_ngo(data):
    if isinstance(data, str):
        return data.replace(' ngo ', ' non-governmental organization ')
    elif isinstance(data, list):
        return [replace_ngo(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_ngo(value) for key, value in data.items()}
    else:
        return data

# FOR IGO

def replace_igo(data):
    if isinstance(data, str):
        return data.replace(' igo ', ' intergovernmental organization ')
    elif isinstance(data, list):
        return [replace_igo(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_igo(value) for key, value in data.items()}
    else:
        return data

# FOR PER

def replace_per(data):
    if isinstance(data, str):
        return data.replace(' per ', ' person ')
    elif isinstance(data, list):
        return [replace_per(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_per(value) for key, value in data.items()}
    else:
        return data

# FOR ORG_GOV

def replace_orggov(data):
    if isinstance(data, str):
        return data.replace(' gov_org ', ' governmental organizzation ')
    elif isinstance(data, list):
        return [replace_orggov(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_orggov(value) for key, value in data.items()}
    else:
        return data

# FOR ORG

def replace_org(data):
    if isinstance(data, str):
        return data.replace(' org ', ' organizzation ')
    elif isinstance(data, list):
        return [replace_org(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_org(value) for key, value in data.items()}
    else:
        return data

# FOR MISC misc => miscellaneous entities

def replace_misc(data):
    if isinstance(data, str):
        return data.replace(' misc ', ' miscellaneous entities ')
    elif isinstance(data, list):
        return [replace_misc(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_misc(value) for key, value in data.items()}
    else:
        return data


# MODIFYING

with open('/content/validation.json', 'r') as f:
    data = json.load(f)

data = replace_gpe0(data)
data = replace_gpe1(data)
data = replace_gpe2(data)
data = replace_gpe0x(data)
data = replace_gpe1x(data)
data = replace_gpe2x(data)
data = replace_in0(data)
data = replace_loc(data)
data = replace_ngo(data)
data = replace_igo(data)
data = replace_per(data)
data = replace_orggov(data)
data = replace_org(data)
data = replace_misc(data)


with open('/content/outputfin.json', 'w') as f:
    json.dump(data, f, indent=4)
