import os
import json
import argparse

import numpy as np
from tqdm import tqdm
from prepro_util import find_span_from_text

title_s = "<title>"
title_e = "</title>"

def save(data, dir_name, data_type):
    if not os.path.isdir(os.path.join('data', dir_name)):
        os.makedirs(os.path.join('data', dir_name))

    file_path = os.path.join('data', dir_name, '{}.json'.format(data_type))
    with open(file_path, 'w') as f:
        print ("Saving {}".format(file_path))
        json.dump({'data': data}, f)

def title_match(evidences, title):
    for eve in evidences:
        if title.strip().find(eve.strip()) >=0:
            return True
    return False 
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='hotpotqa')
    parser.add_argument('--task', type=str, default="hotpot-all")
    parser.add_argument('--data_type', type=str)
    args = parser.parse_args()

    if args.task == 'convert':
        if args.data_type == '2wiki':
            training_data = load_hotpot(args, '2wiki_train')
            save(training_data, 'hotpot-all', '2wiki_train')
            dev_data = load_hotpot(args,  '2wiki_dev')
            save(dev_data, 'hotpot-all', '2wiki_dev')
        elif args.data_type == 'hotpot':
            training_data = load_hotpot(args, 'train')
            save(training_data, 'hotpot-all', 'train')
            dev_data = load_hotpot(args,  'dev_distractor')
            save(dev_data, 'hotpot-all', 'dev')
    else:
        raise NotImplementedError()

def load_hotpot(args, data_type, only_bridge=False, only_comparison=False,only_sf=False, only_gold=False):
    if args.data_type !='2wiki':
        with open(os.path.join(args.data_dir, "hotpot_{}_v1.json".format(data_type)), 'rb') as f:
            data = json.load(f)
    else:
        with open(os.path.join(args.data_dir, "{}.json".format(data_type)), 'rb') as f:
            data = json.load(f)
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    data_list = []
    n_paras = []
    n_gold_paras = []
    n_paras_with_answer = []
    n_sents = []
    n_answers = []
    no_answer = 0
    acc_list = {'overall': [], 'comparison':[], 'bridge':[]}

    for article_id, article in tqdm(enumerate(data)):
        if article['type'] == 'comparison' or  article['type'] == 'bridge_comparison':
            continue
        paragraphs = article['context']
        sfs = [(_process_sent(t), s) for t, s in article['supporting_facts']]
        evidences = set([ele[0] for ele in article['supporting_facts']])
        
        question = article['question']
        answer = article['answer'].strip()

        
        processed_contexts = []
        for para_idx, para in enumerate(paragraphs):
            title = _process_sent(para[0])
            content = para[1]
            if title_match(evidences, title) == True:#筛选
            #if True:
                answers = []
                context = ["{} {} {}".format(title_s, title.lower().strip(), title_e)]
                #context = []
                offset = len(context[0]) + 1
                #offset = 0
                for sent_idx, sent in enumerate(content):
                        is_sf = (title, sent_idx) in sfs
                        if only_sf and not is_sf:
                            continue
                        context.append(sent.lower().strip())
                processed_contexts.append(" ".join(context))
        processed_contexts = [' '.join(processed_contexts)]
        #print(processed_contexts)
        #将多段合成一段
        para_with_sf = set()
        contexts_list, answers_list = [], []
        for para_idx, para in enumerate(paragraphs):
            title = _process_sent(para[0])
            content = para[1]
            if title_match(evidences, title) == True:
            #if True:
                answers = []
                contexts = ["{} {} {}".format(title_s, title.lower().strip(), title_e)]
                #contexts = []
                offset = len(contexts[0]) + 1
                #offset = 0
                if only_gold and title not in [t for t, _ in sfs]:
                    continue

                for sent_idx, sent in enumerate(content):
                    is_sf = (title, sent_idx) in sfs
                    if only_sf and not is_sf:
                        continue
                    contexts.append(sent.lower().strip())
                    if is_sf:
                        para_with_sf.add(para_idx)
                        if answer in ['yes', 'no']:
                            answers.append({'text': answer, 'answer_start': -1})
                        elif answer.lower() in contexts[-1]:
                            assert contexts[-1] == sent.lower().strip()
                            #curr_answers = find_span_from_text(contexts[-1], contexts[-1].split(' '), answer.lower())
                            curr_answers = find_span_from_text(processed_contexts[-1], contexts[-1].split(' '), answer.lower())
                            #for i, curr_answer in enumerate(curr_answers):
                                #curr_answers[i]['answer_start'] += offset
                            answers += curr_answers
                    offset += len(contexts[-1]) + 1

                if len(contexts)>1:
                    n_sents.append(len(contexts))
                    context = " ".join(contexts)
                    contexts_list.append(context)
                    answers_list.append(answers)

        #assert len(para_with_sf)>1
        #assert len(contexts_list)>1

        if only_sf:
            merged_context = ""
            merged_answers = []
            offset = 0
            for (context, answers) in zip(contexts_list, answers_list):
                for i, a in enumerate(answers):
                    answers[i]['answer_start'] += len(merged_context)
                merged_context += context + " "
                merged_answers += answers
            contexts_list, answers_list = [merged_context], [merged_answers]

        assert len(contexts_list)==len(answers_list)
        n_paras.append(len(contexts_list))
        n_gold_paras.append(len(para_with_sf))
        n_paras_with_answer.append(len([a for a in answers_list if len(a)>0]))

        for (context, answers) in zip(contexts_list, answers_list):
            for a in answers:
                if a['text'] not in ['yes', 'no']:
                    assert a['text'] == processed_contexts[0][a['answer_start']:a['answer_start']+len(a['text'])]
        #去掉没找到答案的
        total = 0
        for answers in answers_list:
            if len(answers) !=0:
                total += 1
        if total ==0:
            continue
        n_answers.append(sum([len(answers) for answers in answers_list]))
        if n_answers[-1] == 0:
            no_answer += 1
        paragraph = {
                'context': " ".join(contexts_list),
                'qas': [{
                    'final_answers': [answer],
                    'question': question,
                    'answers': answers_list,
                    'id': article['_id'],
                    'type': article['type']
                }]
            }
        data_list.append({'title': '', 'paragraphs': [paragraph]})

    print ("We have {}/{} number ({} with no answer) of HOTPOT examples!".format(len(data_list), len(data), no_answer))
    print ("On average, # paras = %.2f (%.2f gold and %.2f with answer ) / # sentences = %.2f / # answers = %.2f" % \
           (np.mean(n_paras), np.mean(n_gold_paras), np.mean(n_paras_with_answer), np.mean(n_sents), np.mean(n_answers)))

    return data_list


if __name__ == '__main__':
    main()

#python convert_hotpot2squad_new.py --data_dir dataset --task hotpot-all