import json
from collections import OrderedDict
input_file1='../data/decomposed-predictions/dev_b_2_output_nbest_file.json'
input_file2='../data/decomposed-predictions/dev_output_nbest_file.json'

def find_max(dict_data):
    result=OrderedDict()
    for key,value in dict_data.items():
        max=float(value[0]['probability'])
        result[key] = value[0]
        for item in value:
            if float(item['probability'])>max:
                max=float(item['probability'])
                result[key]=item
    return result

def find_max_in_dicts(one_hop_dict,multi_dict):
    result_dict=OrderedDict()
    for key in one_hop_dict:
        if key in multi_dict:
            if float(multi_dict[key]['probability'])>float(one_hop_dict[key]['probability']):
                result_dict[key]=multi_dict[key]
            else:
                result_dict[key] = one_hop_dict[key]
        else:
            result_dict[key] = one_hop_dict[key]
    return result_dict


with open(input_file1, "r", encoding='utf-8') as reader:
    dict1 = json.load(reader)
with open(input_file2, "r", encoding='utf-8') as reader:
    dict2 = json.load(reader)
new_dict_1=find_max(dict1)
new_dict_2=find_max(dict2)
result=find_max_in_dicts(new_dict_2,new_dict_1)
with open('final.json', 'w') as f:
    json.dump(result, f)
print(result)