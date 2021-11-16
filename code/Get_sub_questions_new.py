import nltk
from nltk import Tree
import benepar
import spacy
import itertools
from nltk.tree import ParentedTree

nlp = spacy.load("en_core_web_sm")
parser = benepar.Parser("./benepar_en3")

def Extract_Clauses(parse_tree,sentence):#抽取出从句
    subtexts = []
    Labels = ['NP','NML','S','SBAR','SQ','SINV']
    Clauses = ['NML','S','SBAR','SQ','SINV']
    Phrases = ['NP']
    parse_tree = ParentedTree.convert(parse_tree)
    for subtree in parse_tree.subtrees():
        parent = subtree.parent()
        #print(parent)
        subtext = ' '.join(subtree.leaves())
        subtext_length = len(subtext.split())
        sentence_length = len(sentence.split())
        if subtree.label() == 'TOP':
            new_sentence = ' '.join(subtree.leaves())
        if subtree.label() in Labels and (subtext_length >=3 and sentence_length - subtext_length >=3):
            if parent == None:
                print("1.插入子节点", subtext,subtree.label())
                subtexts.append(subtext)
                break
            elif subtree.label() in Clauses:#假如现在是从句
                if parent.label() in Clauses:#若父节点也是从句
                    parenttext = ' '.join(parent.leaves())
                    parenttext_length = len(parenttext.split())
                    if parenttext_length >=3 and sentence_length - parenttext_length >=3:
                        print("2.插入父节点",parenttext,parent.label())
                        subtexts.append(parenttext)
                        break
                    else:
                        print("3.插入子节点", subtext,subtree.label())
                        subtexts.append(subtext)
                        break
                else:
                    print("4.插入子节点", subtext,subtree.label(),"父节点标签:",parent.label())
                    subtexts.append(subtext)
                    break
            elif subtree.label() in Phrases and parent.label() in Clauses:
                parenttext = ' '.join(parent.leaves())
                parenttext_length = len(parenttext.split())
                if parenttext_length >=3 and sentence_length - parenttext_length >=3:
                    print("5.插入父节点",parenttext,parent.label())
                    subtexts.append(parenttext)
                    break
                else:
                    print("9.插入子节点", subtext,subtree.label(),"父节点标签:",parent.label())
                    subtexts.append(subtext)
                    break
            elif subtree.label() in Phrases and parent.label() in Phrases:
                parenttext = ' '.join(parent.leaves())
                parenttext_length = len(parenttext.split())
                if parenttext_length >=3 and sentence_length - parenttext_length >=3:
                    print("6.插入父节点",parenttext,parent.label())
                    subtexts.append(parenttext)
                    break
                else:
                    print("7.插入子节点", subtext,subtree.label())
                    subtexts.append(subtext)
                    break
            else:
                print("8.插入子节点", subtext,subtree.label(),"父节点标签:",parent.label())
                subtexts.append(subtext)
                break
                
    for i in reversed(range(len(subtexts)-1)):
        subtexts[i] = subtexts[i][0:subtexts[i].find(subtexts[i+1])]
        if subtexts[i] =='':
            del subtexts[i]
    return new_sentence,subtexts

def Clauses_Extraction(sentence):
    parse_tree = parser.parse(sentence)
    Phrases = Extract_Clauses(parse_tree,sentence)
    return Phrases

def generate_trees(root):#分解并列句
    """
    Yield all conjuncted variants of subtrees that can be generated from the given node.
    A subtree here is just a set of nodes.
    """
    prev_result = [root]
    if not root.children:
        yield prev_result
        return
    children_deps = {c.dep_ for c in root.children}
    if 'conj' in children_deps:
        # generate two options: subtree without cc+conj, or with conj child replacing the root
        # the first option:
        good_children = [c for c in root.children if c.dep_ not in {'cc', 'conj'}]
        for subtree in combine_children(prev_result, good_children):
            yield subtree 
        # the second option
        for child in root.children:
            if child.dep_ == 'conj':
                for subtree in generate_trees(child):
                    yield subtree
    else:
        # otherwise, just combine all the children subtrees
        for subtree in combine_children([root], root.children):
            yield subtree

def combine_children(prev_result, children):
    """ Combine the parent subtree with all variants of the children subtrees """
    child_lists = []
    for child in children:
        child_lists.append(list(generate_trees(child)))
    for prod in itertools.product(*child_lists):  # all possible combinations
        yield prev_result + [tok for parts in prod for tok in parts]

def Conjunctions_Extraction(sent):
    doc = nlp(sent)
    sentences = list(doc.sents)
    sub_sentences = []
    for sentence in sentences:
        for tree in generate_trees(sentence.root):
            subtext = ' '.join([token.text for token in sorted(tree, key=lambda x: x.i)])
            if len(sub_sentences)==2:
                sub_sentences[0] = sub_sentences[0] +" "+subtext
                sub_sentences[1] = sub_sentences[1] +" "+subtext
            else:
                sub_sentences.append(subtext)
    return sub_sentences