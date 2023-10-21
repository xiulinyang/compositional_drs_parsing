import collections
from nltk import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from list_manipulator import manipulate, overlap
from nltk.corpus import stopwords
from sbn_spec_release5 import (
    SBNError,
    AlignmentError,
)
from tqdm import tqdm
from pathlib import Path
import penman
from penman.graph import Graph
from penman.codec import PENMANCodec


def get_edge_info(graph, pre_map):
    '''
    :graph: the built SBNGraph
    :return: [Tuple(current_id, to_id)]the edge information for the graph G
    '''
    edges_info = []
    for edge_info in graph.edges.data(True):
        current_id = pre_map[edge_info[0][0]] + str(edge_info[0][1])
        to_id = pre_map[edge_info[1][0]] + str(edge_info[1][1])
        edges_info.append((current_id, to_id))
    return edges_info


def get_node_info(graph, pre_map, raw_text_path):
    '''
    :param graph: SBNGraph
    :param pre_map: node and id map
    :return: comment_node: List[String] that denote the node ids that have comments;
    comment_node_pair_info: Dict{comment: (node info)}
    cleaned_comment_list: List[Tuple(token, id)]
    '''
    nodes_info = []
    comments = []
    comment_taken = []
    comment_node_pair_info = {}  # TODO: the node info can be used to get a more fine-grained token-node alignment
    comment_node_pair = {}
    for info in graph.nodes.data(True):
        if 'comment' not in list(info[1].keys()) and info[1]['token']=='B-0':
            info[1]['comment'] ='START0'
        if 'comment' in list(info[1].keys()):
            node_type = info[0][0]
            var_id = pre_map[node_type] + str(info[0][1])
            token_node = info[1]['token']
            comment = info[1]['comment']
            nodes_info.append((var_id, token_node, comment))

            if comment != None:
                comments.append(comment)
                if comment not in comment_taken:
                    comment_node_pair[comment] = [var_id]
                    comment_node_pair_info[comment] = [(var_id, token_node)]
                    comment_taken.append(comment)
                else:
                    comment_node_pair_info[comment].append((var_id, token_node))
                    comment_node_pair[comment].append(var_id)
    cleaned_comment_list = [x if 'every' not in x else x.replace("every", "every ", 1).split() for x in
                            tokenize(Path(raw_text_path).read_text())]
    cleaned_comment_list = flatten_list(cleaned_comment_list)
    cleaned_comment_list.insert(0, 'START')
    cleaned_comment_list = [(x, i) for i, x in enumerate(cleaned_comment_list)]
    return nodes_info, comment_node_pair, cleaned_comment_list


def flatten_list(nested_list):
    """
    Flatten the given nested list if there are nested lists, otherwise return the list.
    """
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def find_kid_node(var_id, edges_info, possible_nodes, comment_node):
    target_n_kid = [x[1] for x in edges_info if x[0] == var_id]
    if len(target_n_kid) > 0:
        possible_nodes.append((var_id, target_n_kid[0]))
        if target_n_kid[0] not in comment_node:
            find_kid_node(target_n_kid[0], edges_info, possible_nodes, comment_node)
        else:
            return possible_nodes
    else:
        raise AlignmentError(
            "The node is isolate!"
        )


def group_nodes_for_alignment(nodes_info, edges_info, comment_node_pair):
    comment_node = [x for y in list(comment_node_pair.values()) for x in y]
    for n in nodes_info:
        if n[-1] == None:
            possible_nodes = []
            def find_node(var_id1):
                target_n = [x[0] for x in edges_info if x[1] == var_id1]
                target_n= sorted(target_n, key=lambda x: 'b' in x)
                if len(target_n) > 0:
                    possible_nodes.append((var_id1, target_n[0]))
                    if target_n[0] not in comment_node:
                        find_node(target_n[0])
                    else:
                        return possible_nodes
                else:
                    target_n_kid = [x[1] for x in edges_info if x[0] == var_id1]
                    if len(target_n_kid) > 1:
                        target_n_kid_info = [x[1] for x in edges_info if x[1] in target_n_kid]
                        counts = collections.Counter(target_n_kid_info)

                        target_n_kid_new = sorted(target_n_kid, key=lambda x: -counts[x])
                        possible_nodes.append((var_id1, target_n_kid_new[0]))
                        if target_n_kid_new[0] not in comment_node:
                            find_node(target_n_kid_new[0])
                        else:
                            return possible_nodes
                    elif len(target_n_kid) == 1:
                        possible_nodes.append((var_id1, target_n_kid[0]))
                        return possible_nodes
                    else:
                        raise AlignmentError(
                            'The node is isolate!'
                        )

            find_node(n[0])
            if not possible_nodes:
                continue
            else:
                for comment_token, aligned_nodes in comment_node_pair.items():
                    if possible_nodes[-1][-1] in aligned_nodes:
                        possible_nodes = [x for y in possible_nodes for x in y]
                        possible_nodes = list(dict.fromkeys(possible_nodes))
                        aligned_nodes.append(possible_nodes)
    for k, v in comment_node_pair.items():
        comment_node_pair[k] = list(set(flatten_list(v)))
    return comment_node_pair


def split_list_using_separators(main_list):
    result = []
    current_sublist = []
    current_sublist_element = []
    result_element = []
    # print(proposition_list)
    c = 0
    box_c = 0
    for element in main_list:
        if len(element) == 2:
            box_c += 1
            if current_sublist:
                result.append(current_sublist)
                result_element.append(current_sublist_element)
            else:
                result.append([])
                result_element.append([])
            element.append(box_c)
            current_sublist = [element]
            current_sublist_element = [element]  # Clear the current sublist to start building a new one
        else:
            current_sublist.append(c)
            current_sublist_element.append((element, c))
            c += 1

    # Append the last sublist, if any
    if current_sublist:
        result.append(current_sublist)
    if current_sublist_element:
        result_element.append(current_sublist_element)

    return result, result_element


def scope_assignment(proposition_list, result, result_element):
    if proposition_list:
        for id, index in proposition_list:
            # id: the node id where the proposition lies
            # index: the index after proposition
            context_id = result.index([x for x in result if id in x][0])

            if int(index) > 0:
                # if proposition >n
                assert result_element[context_id + int(index)][0][0] == 'CONTINUATION' and \
                       result_element[context_id + int(index)][0][1] == '-0'
                result_element[context_id + int(index)][0] = ['Proposition', id, result[context_id + int(index)][0][2]]
            elif int(index) < 0:
                # if proposition <n, move the CONTINUATION BOX to the target discourse unit with a new name 'Proposition'
                # id is the node id of the predicate
                assert result_element[context_id][0][0] == 'CONTINUATION' and result_element[context_id][0][1] == '-0'
                useless_continuation = result_element[context_id].pop(0)
                if type(result[context_id + int(index)][0]) == int:
                    # if the argument does not have any other discourse connective
                    result_element[context_id + int(index)].insert(0, ['Proposition', id, useless_continuation[-1]])
                else:
                    # otherwise
                    result_element.insert(context_id + int(index), ['Proposition', id, useless_continuation[-1]])

            else:
                raise SBNError('the graph might not be right.')
    for i, unit in enumerate(result_element):
        if i == 0 and len(unit) == 0:
            unit.insert(0, ['ROOT', '-0', 0])
        if len(unit) > 0 and len(unit[0][0].split('.')) == 3:
            unit.insert(0, ['ROOT', '-0', 0])
    return result_element


def alignment(comment_node_pair, cleaned_comment_list, nodes_info, edges_info):
    '''

    :param comment_node_pair_info: the dictionary of comment and its corresponding node_id and node info pair
    :param cleaned_comment_list: the tokenized sentence
    :return:
    '''

    node_id_pair = {x[0]: x[1] for x in nodes_info}
    indx_token_pair = {}
    for node in nodes_info:
        if node[0] == 'b0':
            indx_token_pair[node[0]] = 'START'
        else:
            indx_token_pair[node[0]] = node[1].split('.')[0]

    alignment_list = []
    for tok, nodes in comment_node_pair.items():
        tok = tokenize(tok[:-1])
        tokens_without_sw = [word for word in tok]
        alignment = {}
        # tokens_without_sw.sort(key=len, reverse=True) # choose the longest word as the lexical node
        if len(nodes) == 1:
            aligned_token = \
            sorted(tokens_without_sw, key=lambda x: (indx_token_pair[nodes[0]] in x, len(x)), reverse=True)[0]
            aligned_node = [t for t in cleaned_comment_list if t[0] == aligned_token]
            if aligned_node:
                alignment['token_id'] = aligned_node[0][1]

            else:
                tgt_token = sorted([x[0] for x in cleaned_comment_list], key=lambda x: overlap_count(x, aligned_token),
                                   reverse=True)[0]
                aligned_node = [t for t in cleaned_comment_list if t[0] == tgt_token]
                alignment['token_id'] = aligned_node[0][1]
            alignment['node_id'] = nodes[0]
            alignment_list.append(alignment)
            cleaned_comment_list.remove(aligned_node[0])
        elif len(nodes) > 1:
            related_edges = [edge for edge in edges_info if all(elem in nodes for elem in edge)]
            related_nodes = set([y for x in related_edges for y in x])
            to_nodes = [x[1] for x in related_edges]
            parent_node = [x for x in related_nodes if x not in to_nodes]
            assert len(parent_node) > 0
            aligned_token = \
            sorted(tokens_without_sw, key=lambda x: (indx_token_pair[parent_node[0]] in x, len(x)), reverse=True)[0]
            aligned_node = [t for t in cleaned_comment_list if t[0] == aligned_token]
            if aligned_node:
                alignment['token_id'] = aligned_node[0][1]
            else:
                tgt_token = sorted([x[0] for x in cleaned_comment_list], key=lambda x: overlap_count(x, aligned_token),
                                   reverse=True)[0]
                aligned_node = [t for t in cleaned_comment_list if t[0] == tgt_token]
                alignment['token_id'] = aligned_node[0][1]
            alignment['node_id'] = nodes
            nodes_details = [v for k, v in node_id_pair.items() if k in nodes]

            assert len(nodes) == len(nodes_details)
            best_pair = find_the_best_overlap_pair(nodes_details, tokens_without_sw)
            lexical_node = [k for k, v in node_id_pair.items() if v == best_pair[0] and k in nodes]
            alignment['lexical_node'] = lexical_node
            if len(lexical_node) > 1:
                assert node_id_pair[lexical_node[0]] == node_id_pair[lexical_node[-1]]
            alignment['lexical_node'] = alignment['lexical_node'][0]
            alignment_list.append(alignment)
            cleaned_comment_list.remove(aligned_node[0])
        else:
            raise AlignmentError(
                'The token does not have a corresponding node!'
            )

    return alignment_list


def find_the_best_overlap_pair(nodes_list, comment_list):
    max_overlap = 0
    best_pair = None
    for item1 in nodes_list:
        for item2 in comment_list:
            overlap = len(set(item1).intersection(item2))
            if overlap > max_overlap:
                max_overlap = overlap
                best_pair = (item1, item2)
    if max_overlap < 2:
        target_comment = sorted(comment_list, key=lambda x: (len(x)), reverse=True)[0]
        target_node = sorted(nodes_list, key=lambda x: overlap_count(x, target_comment), reverse=True)[0]
        best_pair = (target_node, target_comment)
    return best_pair


def overlap_count(element, given_element):
    return len(set(element).intersection(given_element))


def tokenize(input):
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(input)


def get_split_list(split_dir):
    split = open(split_dir, 'r').readlines()
    return [x.strip() for x in split]


def generate_gold_split(split_dir, output_file, version):
    codec = PENMANCodec()
    root_dir = f'ud-boxer/data/pmb-{version}.0.0/data/en'
    with open(split_dir, 'r') as split, open(output_file, 'w') as split_out:
        split_dir = [x.strip() for x in split.readlines()]
        for i, dir in tqdm(enumerate(split_dir)):
            final_dir = f'{root_dir}/{dir}/en.drs.penman'

            penman_string = Path(final_dir).read_text().strip()

            # penman_triples = penman.decode(penman_string).triples

            # new_triples = []
            # for tri in set(penman_triples):
            #     if ":" in tri[2]:
            #         new_triples.append(tri)
            #     elif "/" in tri[2]:
            #         new_triples.append(tri)
            #     elif len(tri[2].split()) > 1:
            #         node = '_'.join(tri[2][1:-1].split())
            #         new_triples.append((tri[0], tri[1], node))
            #     elif tri[2] == "_":
            #         node = 'entity.n.01'
            #         new_triples.append((tri[0], tri[1], node))
            #     else:
            #         node = tri[2].replace('"', '')
            #         new_triples.append((tri[0], tri[1], node))
            #     for tri in set(penman_triples):
            #         if len(tri[2].split('.')) == 3 or 'box' in tri[2]:
            #             node = tri[2].replace('"', '')
            #             new_triples.append((tri[0], tri[1], node))
            #         elif tri[2][-1].isnumeric() and tri[2][0].isalpha():
            #             print(tri)
            #             new_triples.append(tri)
            #         else:
            #             if tri[2][0] != '\"':
            #                 new_node = f'\"{tri[2]}\"'
            #                 new_triples.append((tri[0], tri[1], new_node))
            #             else:
            #                 new_triples.append(tri)
            # new_triples = sorted(new_triples, key=lambda x: x[0])
            # new_penman_string = codec.encode(Graph(new_triples))
            if 'gold' in dir:
                split_out.write(penman_string)
                split_out.write('\n\n')
            else:
                gold_dev = ' '.join([x.strip() for x in new_penman_string.split('\n')])
                split_out.write(gold_dev)
                split_out.write('\n\n')

