from pathlib import Path
from penman.codec import PENMANCodec
import spacy
from spacy.tokens import Doc
from sbn_spec import SBNSpec
def customtokenizer(text):
    return Doc(nlp.vocab, text.split())

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = customtokenizer
def find_reentrancies_gold(split_path, gold_penman, file=None):
    root = 'ud-boxer/data/pmb-4.0.0/data/en/gold/'
    sents_reentrancy =[]
    path_reentrancy = []
    penman_reentrancy =[]
    split_path = Path(split_path).read_text().strip().split('\n')
    print(len(split_path))
    gold_penman = Path(gold_penman).read_text().strip().split('\n\n')
    print(len(gold_penman))
    for i, penman_string in enumerate(gold_penman):
        codec = PENMANCodec()
        p = split_path[i]
        if file=='train':
            root = 'ud-boxer/data/pmb-4.0.0/data/en/'
            raw = ' '.join(Path(root+p[:-5]+'/en.raw').read_text().split())
        else:
            raw = ' '.join(Path(root+p+'/en.raw').read_text().split())
        triples = codec.decode(penman_string).triples
        to_node_info = [x[2] for x in triples if x[1] !=':member' and x[1][1:] not in SBNSpec.INVERTIBLE_ROLES2 and x[2][0]=='s' and x[2][1].isnumeric()]

        if len(set(to_node_info))<len(to_node_info):
            print(to_node_info)
            sents_reentrancy.append(raw)
            path_reentrancy.append(p)
            penman_reentrancy.append((penman_string, to_node_info))

    return sents_reentrancy, path_reentrancy, penman_reentrancy

def get_predictions(gold_path, pred_path, pred_file):
    pred_path = Path(pred_path).read_text().split('\n')
    pred_file = Path(pred_file).read_text().split('\n\n')
    pred_path_penman = {x[len('gold/'):-len('/scopeless4')]:y for x,y in zip(pred_path, pred_file)}
    pred_penman_list =[]
    for p in gold_path:
        pred_penman_list.append(pred_path_penman[p])
    return pred_penman_list

def classify_reentrancy(path, penman, sent):
    coref =[]
    time = []
    coordination =[]
    verbalisation =[]
    control=[]
    relative = []
    others =[]


    for i, s in enumerate(sent):
        parsed_sent = nlp(s)
        text = [x.text for x in parsed_sent]
        dep_edge = [x.tag_ for x in parsed_sent]
        ner = [x.label_ for x in parsed_sent.ents]

        if 'DATE' in ner or 'tomorrow' in text:
            time.append((path[i], s, penman[i]))
        elif 'CC' in dep_edge:
            coordination.append((path[i], s, penman[i]))
        elif 'TO' in dep_edge:
            control.append((path[i], s, penman[i]))
        elif 'favorite' in text or 'novel-based' in text or 'VBN' in dep_edge:
            verbalisation.append((path[i], s, penman[i]))
        elif 'NN WP' in ' '.join(dep_edge):
            relative.append((path[i], s, penman[i]))
        elif 'PRP' in dep_edge or 'PRP$' in dep_edge:
            coref.append((path[i], s, penman[i]))
        else:
            others.append((path[i], s, penman[i]))

    return coref, time, coordination, verbalisation, control, relative, others
train4 = "ud-boxer/ud_boxer/gold4/en_train.txt"
train4_gold = 'ud-boxer/ud_boxer/gold4/gold_train.txt'
dev4 = "ud-boxer/ud_boxer/data_split/gold4/en_dev.txt"
dev4_gold = 'ud-boxer/ud_boxer/gold4/gold_dev.txt'
test4 = "ud-boxer/ud_boxer/data_split/gold4/en_test.txt"
test4_gold = 'ud-boxer/ud_boxer/gold4/gold_test.txt'
eval4 = "ud-boxer/ud_boxer/data_split/gold4/en_eval.txt"
eval4_gold = 'ud-boxer/ud_boxer/gold4/gold_eval.txt'

dev4_pred_path = 'ud-boxer/ud_boxer/data_split/scopeless4/en_dev.txt'
test4_pred_path = 'ud-boxer/ud_boxer/data_split/scopeless4/en_test.txt'
eval4_pred_path = 'ud-boxer/ud_boxer/data_split/scopeless4/en_eval.txt'

dev4_pred = 'ud-boxer/ud_boxer/seq2seqoutput/4/udboxer_dev_gold.txt'
test4_pred = 'ud-boxer/ud_boxer/seq2seqoutput/4/udboxer_test_gold.txt'
eval4_pred = 'ud-boxer/ud_boxer/seq2seqoutput/4/udboxer_eval_gold.txt'

# train4_sents, train4_paths, train4_penman=find_reentrancies_gold(train4,train4_gold, 'train')
dev4_sents, dev4_paths, dev4_penman=find_reentrancies_gold(dev4,dev4_gold)
test4_sents, test4_paths, test4_penman=find_reentrancies_gold(test4,test4_gold)
eval4_sents, eval4_paths, eval4_penman=find_reentrancies_gold(eval4,eval4_gold)
# print(dev4_paths)
dev4_pred_penman = get_predictions(dev4_paths, dev4_pred_path, dev4_pred)
test4_pred_penman = get_predictions(test4_paths, test4_pred_path, test4_pred)
eval4_pred_penman = get_predictions(eval4_paths, eval4_pred_path, eval4_pred)

pmb4_sents = dev4_sents+test4_sents+eval4_sents
pmb4_paths = dev4_paths+test4_paths+eval4_paths
pmb4_penman = dev4_penman+test4_penman+eval4_penman
pred4_penman = dev4_pred_penman+test4_pred_penman+eval4_pred_penman
# pmb4_sents = dev4_sents+test4_sents
# pmb4_paths = dev4_paths+test4_paths
# pmb4_penman = dev4_penman+test4_penman
# pred4_penman = dev4_pred_penman+test4_pred_penman


coref, time, coordination, verbalisation, control, relative, others = classify_reentrancy(pmb4_paths, pmb4_penman, pmb4_sents)

with open('pmb4_reentrancy.txt', 'w') as reen:
    for x in pmb4_sents:
        reen.write(f'{x}\n')

with open('pmb4_reentrancy_penman.txt', 'w') as penman_re:
    for x, _ in pmb4_penman:
        # penman_re.write(f'{y}\n')
        penman_re.write(f'{x}\n\n')

with open('pmb4_reentrancy_penman_pred_ud.txt', 'w') as penman_pred:
    for x in pred4_penman:
        # penman_re.write(f'{y}\n')
        penman_pred.write(f'{x}\n\n')

with open('pmb4_reentrancy_path.txt', 'w') as re_path:
    for x in pmb4_paths:
        re_path.write(f'{x}\n')


print(len(pmb4_sents))

