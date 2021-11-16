import json
import datetime
import string
import argparse
from Get_sub_questions_new import Clauses_Extraction, Conjunctions_Extraction

punctuation_string = string.punctuation#remove the punctuation
parser = argparse.ArgumentParser("Question_Decomposition")
parser.add_argument("--data_type", type=str, default="dev")
args = parser.parse_args()
data_type = args.data_type

Decomposed_Questions = []

if data_type == '2wiki_dev':
    input_data = json.load(open('data/hotpot-all/2wiki_dev.json', 'rb'))['data']
elif data_type =='2wiki_train':
    input_data = json.load(open('data/hotpot-all/2wiki_train.json', 'rb'))['data']
elif data_type =='dev':
    input_data = json.load(open('data/hotpot-all/dev.json', 'rb'))['data']
elif data_type =='train':
    input_data = json.load(open('data/hotpot-all/train.json', 'rb'))['data']

for i, data in enumerate(input_data):
    _id = data['paragraphs'][0]['qas'][0]['id']
    print(i, '/', len(input_data))
    starttime = datetime.datetime.now()
    question = data['paragraphs'][0]['qas'][0]['question']
    Sub_Questions = []
    new_sentence, Clauses = Clauses_Extraction(question)
    for i, phrase in enumerate(Clauses):
        phrase = ' '.join(phrase.split())
        new_sentence = new_sentence.replace(phrase, "[Answer of Question{}]".format(i))
        Sub_Questions.append(phrase)
    Sub_Questions.append(new_sentence.strip())
    Sub_Questions.append(question)
    if len(Sub_Questions)<3:
        continue
    Decomposed_Questions.append({_id: Sub_Questions})
    print(Sub_Questions)

    endtime = datetime.datetime.now()
    print(endtime - starttime)

print(len(Decomposed_Questions))
file_name =data_type+'_predictions'
with open('./data/{}.json'.format(file_name), 'w') as f:
    json.dump(Decomposed_Questions, f)