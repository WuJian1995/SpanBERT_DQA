import json
from collections import OrderedDict
from run_squad import get_raw_scores,make_eval_dict
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

input_file1='../data/decomposed-predictions/dev_b_2_output_nbest_file.json'
input_file2='../data/decomposed-predictions/dev_output_nbest_file.json'
dev_file='../data/hotpot-all/2wiki_dev.json'
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
                result_dict[key]=multi_dict[key]['text']
            else:
                result_dict[key] = one_hop_dict[key]['text']
        else:
            result_dict[key] = one_hop_dict[key]['text']
    return result_dict


with open(input_file1, "r", encoding='utf-8') as reader:
    dict1 = json.load(reader)
with open(input_file2, "r", encoding='utf-8') as reader:
    dict2 = json.load(reader)
new_dict_1=find_max(dict1)
new_dict_2=find_max(dict2)
preds=find_max_in_dicts(new_dict_2,new_dict_1)
with open(dev_file) as f:
    dataset_json = json.load(f)
eval_dataset = dataset_json['data']
exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
result = make_eval_dict(exact_raw, f1_raw)
logger.info("***** Eval results *****")
for key in sorted(result.keys()):
    logger.info("  %s = %s", key, str(result[key]))
with open('final.json', 'w') as f:
    json.dump(result, f)
print(result)