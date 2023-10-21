from pathlib import Path
from penman.graph import Graph
from penman.codec import PENMANCodec
import copy
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Heuristics that add scope information back')
parser.add_argument('-i', '--input', type=str, required=True,
                    help='the parse file produced by amparser')
parser.add_argument('-o', '--output', type=str, required=True,
                    help = 'the postprocessed output file')

def get_descendants_beyond_children(node, triples):
    children = [(source, role, target) for (source, role, target) in triples if
                source == node]
    children_copy = copy.deepcopy(children)
    triples = [x for x in triples if x not in children_copy]
    for child in children_copy:
        grandchildren, triples = get_descendants_beyond_children(child[2], triples)
        children.extend(grandchildren)
    return children, triples

def get_penman(penman_file):
    return Path(penman_file).read_text()

def add_member(penman_string):
    codec = PENMANCodec()
    triples = codec.decode(penman_string).triples
    instances = [x for x in triples if ':instance' in x]
    id_node_pair = {}
    box_num = []
    nodes_possible_list = []
    constant_node_pair = {}
    from_nodes_list = [x[0] for x in triples if x not in instances]
    membered_node = [x[2] for x in triples if ':member' in x]
    triples_new_name =[]
    for id, ins, node in instances:
        if len(node.split('.')) ==3 or node=="_":
            n = 's'+id[2:]
            nodes_possible_list.append(id)
        elif node =='box':
            n = 'b'+id[2:]
            box_num.append(id)
        elif id in from_nodes_list:
            n ='s'+id[2:]
            nodes_possible_list.append(id)
        else:
            n = 'c'+id[2:]
            constant_node_pair[id] = node
        id_node_pair[id] = n
        triples_new_name.append((n, ins, node))

    if len(box_num)==1:
        for n in nodes_possible_list:
            triples.append((box_num[0], ':member', n))
    else:
        triples_reference = [x for x in triples if ':instance' not in x and id_node_pair[x[2]][0] not in ['b', 'c'] and id_node_pair[x[0]][0]!='c']
        triples_box = sorted(set([x[0] for x in triples if id_node_pair[x[0]][0]=='b' and ':instance' not in x]))
        if len(triples_box)==0:
            print(triples)
        scope_pair = {}
        for i in range(len(triples_box)):
            triple = triples_box[i]
            scope_pair[triple] = []
            children_cluster, rest_triples = get_descendants_beyond_children(triple, triples_reference)
            children_cluster = [x for x in children_cluster if x[2] not in membered_node]
            scope_pair[triple].extend([x[2] for x in children_cluster])
            if children_cluster:
                for child in children_cluster:
                    triples.append((triple, ':member', child[2]))

                    to_remove = [(source, role, target) for (source, role, target) in triples_reference if target == child[2]]
                    for k in to_remove:
                        triples_reference.remove(k)
        membered_nodes = [x[2] for x in triples if x[1]==':member']
        left_nodes = [x for x in nodes_possible_list if x not in membered_nodes]

        # print(triples_box)
        for tri_member in left_nodes:
            triples.append((triples_box[-1], ':member', tri_member))
#TODO remove the c node
    triples_without_constants = []
    # print(constant_node_pair)
    for trip in triples:
        if trip[0] in list(constant_node_pair.keys()) and trip[1]==':instance':
            continue
        else:
            if trip[2] not in list(constant_node_pair.keys()):
                triples_without_constants.append(trip)
            else:
                triples_without_constants.append((trip[0],trip[1],constant_node_pair[trip[2]]))
# remove unnecessary quotation marks
    new_triples = []
    for tri in set(triples_without_constants):
        if ":" in tri[2]:
            new_triples.append(tri)
        elif "/" in tri[2]:
            new_triples.append(tri)
        elif len(tri[2].split())>1:
            new_triples.append(tri)
        elif tri[2] =="_" or ")" in tri[2]:
            node = 'entity.n.01'
            new_triples.append((tri[0], tri[1], node))
        else:
            node = tri[2].replace('"', '')
            new_triples.append((tri[0], tri[1], node))

    #ANA
    pronouns = [x for x in new_triples if '.p.' in x[2] and x[1] == ":instance"]
    ana_pair = {}
    for pro in pronouns:
        noun = pro[2].split('.')
        new_triples.remove(pro)
        new_triples.append((pro[0], pro[1], '.'.join([noun[0], 'n', noun[2]])))

        if pro[2] not in ana_pair:
            ana_pair[pro[2]] = [pro[0]]
        else:
            ana_pair[pro[2]].append(pro[0])

    for k, v in ana_pair.items():
        if len(v) > 1:
            v = sorted(list(v))
            new_triples.append((v[0], ":ANA", v[1]))

    real_new_triples = sorted(new_triples, key=lambda x: x[0], reverse=False)

    return codec.encode(Graph(set(real_new_triples)))



def main(parserout, postprocessed):
    with open(parserout, 'r') as input, open(postprocessed, 'w') as post:
        sbns = [x.strip() for x in input.readlines() if x !='\n']
        for sbn in tqdm(sbns, desc="Postprocess"):
            try:
                penman_string = add_member(sbn)
            except (IndexError) as e:
                penman_string = sbn
                print(e)
            post.write(penman_string)
            post.write('\n\n')



if __name__ == '__main__':
    args = parser.parse_args()
    parseout = args.input
    postprocessed = args.output
    main(parseout, postprocessed)
