import os
import sys
import json
import argparse
import numpy as np
from collections import Counter, defaultdict

from hotpot_evaluate_v1 import normalize_answer, f1_score as hotpot_f1_score

def main():
    parser = argparse.ArgumentParser("Preprocess HOTPOT data")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--task", type=str, default="decompose")
    parser.add_argument("--out_name", default="2wikimultihop_output")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    out_name = args.out_name
    data_type = args.data_type
    new_data_path = "data/decomposed/{}_b.json".format(data_type)
    new_data1_path = "data/decomposed/{}_b.1.json".format(data_type)
    new_data2_path = "data/decomposed/{}_b.2.json".format(data_type)

    if not os.path.isdir(os.path.join('data', 'decomposed-predictions')):
        os.makedirs(os.path.join('data', 'decomposed-predictions'))


    if args.task == "decompose":
        with open(os.path.join('data', 'hotpot-all', '{}.json'.format(data_type)), 'r') as f:
            orig_data = json.load(f)['data']

        with open('data/{}_predictions.json'.format(data_type), 'r') as f:
            result = json.load(f)


        if not os.path.isdir(os.path.join('data', 'decomposed')):
            os.makedirs(os.path.join('data', 'decomposed'))

        prepro(orig_data, result,new_data_path.format(args.data_type),new_data1_path.format(args.data_type),new_data2_path.format(args.data_type))

    elif args.task == 'plug':
        with open('{}/dev_b_1_nbestsize_predictions.json'.format(out_name), 'r') as f:
            out1 = json.load(f)
        with open(new_data1_path, 'r') as f:
            data1 = json.load(f)['data']
        with open(new_data2_path, 'r') as f:
            data2 = json.load(f)['data']

        print (new_data1_path.format(data_type), new_data2_path.format(data_type))

        new_data2 = []
        for i, (d1, d) in enumerate(zip(data1, data2)):
            q = d['paragraphs'][0]['qas'][0]
            assert d1['paragraphs'][0]['qas'][0]['id'] == q['id'] and q['id'] in out1
            qas = []
            prediction = out1[q['id']]
            #print(prediction, q['question'])
            qas.append({'question': q['question'].replace('[answer of question0]', prediction),'id': "{}".format(q['id']),'answers': q['answers']})
            if 'index' in q:
                qas[-1]['index'] = q['index']
            if 'final_answers' in q:
                qas[-1]['final_answers'] = q['final_answers']
                
            new_data2.append({'paragraphs': [{'context': d['paragraphs'][0]['context'], 'qas': qas}]})

        with open(new_data2_path.format(data_type), 'w') as f:
            json.dump({'data': new_data2}, f)

    else:
        raise  NotImplementedError("{} Not Supported".format(args.task))


def prepro(orig_data, results, new_data_path, new_data1_path, new_data2_path):
    new_data0 = []
    new_data1 = []
    new_data2 = []
    k = 0
    for i,datapoint in enumerate(orig_data):
        paragraph = datapoint['paragraphs'][0]['context']
        qa = datapoint['paragraphs'][0]['qas'][0]
        if i%500==0:
            print(i,'/',len(orig_data))
        
        for result in results:
            #question_type = result['question_type']
            keys = list(result.keys())
            if qa['id'] == keys[0]:
                k = k+1
                #得到两个子问题和原问题
                if len(result[qa['id']]) < 3:#if a question is too simple to be decomposed, then treat it as a single-hop question
                    question1 = result[qa['id']][0]#question1 and question2 are the same question
                    question2 = result[qa['id']][0]
                    question = result[qa['id']][1]
                else:
                    question1 = result[qa['id']][0]
                    question2 = result[qa['id']][1]
                    question = result[qa['id']][2]
                assert len(qa['final_answers'])>0
                d0 = {'context': paragraph, 'qas': [{
                    'id': qa['id'], 'question': question.lower(),
                    'final_answers': qa['final_answers'], 'answers': qa['answers']
                }]}
                d1 = {'context': paragraph, 'qas': [{
                    'id': qa['id'], 'question': question1.lower(),
                    'final_answers': qa['final_answers'], 'answers': qa['answers']
                }]}
                d2 = {'context': paragraph, 'qas': [{
                    'id': qa['id'], 'question': question2.lower(),
                    'final_answers': qa['final_answers'], 'answers': qa['answers']
                }]}
                if '[answer of question0]' in question2.lower():#determine the answer order
                    new_data0.append({'paragraphs': [d0]})
                    new_data1.append({'paragraphs': [d1]})
                    new_data2.append({'paragraphs': [d2]})
                else:
                    new_data0.append({'paragraphs': [d0]})
                    new_data1.append({'paragraphs': [d2]})
                    new_data2.append({'paragraphs': [d1]})

    print (len(new_data0), len(new_data1), len(new_data2))

    with open(new_data_path, 'w') as f:
        json.dump({'data': new_data0}, f)
    with open(new_data1_path, 'w') as f:
        json.dump({'data': new_data1}, f)
    with open(new_data2_path, 'w') as f:
        json.dump({'data': new_data2}, f)

def _normalize_answer(text):
    if '<title>' in text:
        text = text.replace('<title>', '')
    if '</title>' in text:
        text = text.replace('</title>', '')

    list1 = ['/title>'[i:] for i in range(len('/title>'))]
    list2 = ['</title>'[:-i] for i in range(1, len('</title>'))] + \
                    ['<title>'[:-i] for i in range(1, len('<title>'))]

    for prefix in list1:
        if text.startswith(prefix):
            text = text[len(prefix):]

    for prefix in list2:
        if text.endswith(prefix):
            text = text[:-len(prefix)]

    if '(' in text and ')' not in text:
        texts = [t.strip() for t in text.split('(')]
        text = texts[np.argmax([len(t) for t in texts])]
    if ')' in text and '(' not in text:
        texts = [t.strip() for t in text.split(')')]
        text = texts[np.argmax([len(t) for t in texts])]

    text = normalize_answer(text)
    return text

def is_filtered(answer_set, new_answer):
    new_answer = _normalize_answer(new_answer)
    if len(new_answer)==0:
        return True
    for answer in answer_set:
        if _normalize_answer(answer) == new_answer:
            return True
    return False

def filter_duplicate(orig_answers):
    answers = []
    for answer in orig_answers:
        if is_filtered([a['text'] for a in answers], answer['text']): continue
        answers.append(answer)
    return answers

def intersection_convert_to_queries(questions, start, end):
    q1, q2 = [], []
    for i, q in enumerate(questions):
        if q==',' and i in [start-1, start, end, end+1]:
            continue
        if i==0:
            if start==0 and q.startswith('wh'):
                status1, status2 = -1, 1
            elif (not q.startswith('wh')) and questions[start].startswith('wh'):
                status1, status2 = 1, 0
            else:
                status1, status2 = 0, 1
        if i<start:
            q1.append(q)
            if status1==0:
                q2.append(q)
        elif i>=start and i<=end:
            if status2==1 and i==start:
                if q=='whose':
                    q1.append('has')
                    continue
                if i>0 and (q in ['and', 'that'] or q.startswith('wh')):
                    continue
            q1.append(q)
            if status2==0:
                q2.append(q)
        elif i>end:
            if i==end+1 and len(q1)>0 and q=='whose':
                q2.append('has')
            elif i!=end+1 or len(q1)==0 or status1==-1  or not (q in ['and', 'that'] or q.startswith('wh')):
                q2.append(q)
    if len(q1)>0 and q1[-1] != '?':
        q1.append('?')
    if len(q2)>0 and q2[-1] != '?':
        q2.append('?')

    return q1, q2

if __name__ == '__main__':
    main()






