import argparse
import json
import logging

from copy import deepcopy
import sbn_spec
from sbn_spec import SBN_NODE_TYPE
import sbn_release5
import sbn_release4
import sbn_scopeless4
import sbn_simplified4
import sbn_simplified5
import sbn_scopeless5
from typing import Generator

from graph_generator_helper import *
import os

parser = argparse.ArgumentParser(description='Generate the DRGs')
parser.add_argument('-s', '--starting_path', type=str, required=True,
                    help='the starting path of the PMB')
parser.add_argument('-f', '--split_file', nargs=4, required=True,
                    help='The paths of four split files. For PMB4, the order is the path of train, dev, test'
                         'and eval, while for PMB5, the order is the path of train, dev, test_standard, and test_long')
parser.add_argument('-v', '--version', type=int, required=True,
                    help = 'the release of PMB')
parser.add_argument('--scopeless', action='store_true',
                    help='to generate scopeless DRGs')
parser.add_argument('--simplified', action='store_true',
                    help='to generate simplified DRGs')
parser.add_argument('--experiment', type=str, required=True,
                    help='the name of the experiments (i.e., scopeless, simplified SBN, or complete'
                         'SBN). After the name of the experiment is specified, the script will generate'
                         'a new foler under the name of the expriment and all generated files will be stored '
                         'in that folder.')
parser.add_argument('--gold', action='store_true',
                    help='to generate gold DRGs')
parser.add_argument('--png', action='store_true',
                    help='to generate png visualizations')
parser.add_argument('-l', '--log_error', action='store_true',
                    help='whether log error message or not')


args= parser.parse_args()

pre_map = {
        SBN_NODE_TYPE.BOX: "b",
        SBN_NODE_TYPE.CONSTANT: "c",
        SBN_NODE_TYPE.SYNSET: "s",
    }

def pmb_generator(
    starting_path: os.PathLike,
    pattern: str,
    # By default we don't want to regenerate predicted output
    exclude: str = "predicted",
    disable_tqdm: bool = False,
    desc_tqdm: str = "",
) -> Generator[Path, None, None]:
    """Helper to glob over the pmb dataset"""
    path_glob = Path(starting_path).glob(pattern)
    return tqdm(
        (p for p in path_glob if exclude not in str(p)),
        disable=disable_tqdm,
        desc=desc_tqdm,
    )

experiment = args.experiment
error = 0
os.makedirs(f'{experiment}/', exist_ok=True)
train_path = f'{experiment}/en_train.txt'
dev_path = f'{experiment}/en_dev.txt'
test_path = f'{experiment}/en_test.txt'
gold_train = f'{experiment}/gold_train.txt'
gold_dev = f'{experiment}/gold_dev.txt'
gold_test = f'{experiment}/gold_test.txt'
if args.version==4:
    other_path = f'{experiment}/en_eval.txt'
    gold_other = f'{experiment}/gold_eval.txt'
elif args.version==5:
    other_path = f'{experiment}/test_long.txt'
    gold_other = f'{experiment}/gold_test_long.txt'
train_split = get_split_list(f'{args.split_file[0]}')
dev_split = get_split_list(f'{args.split_file[1]}')
test_split = get_split_list(f'{args.split_file[2]}')
other_split = get_split_list(f'{args.split_file[3]}')
split_all = train_split+dev_split+test_split+other_split

with open(train_path, 'w') as train_correct,open(dev_path, 'w') as dev_correct, open(other_path, 'w') as other_correct, open(test_path, 'w') as test_correct:
    for file in tqdm(split_all, desc='Processing SBNs'):
        root_path = args.starting_path
        filepath = Path(os.path.join(root_path, file, 'en.drs.sbn'))
        try:
            os.makedirs(f'{filepath.parent}/{experiment}/', exist_ok=True)
            raw_path = str(filepath).replace('drs.sbn', 'raw')
            if args.version==4:
                if args.simplified:
                    G = sbn_simplified4.SBNGraph().from_path(filepath)
                    if not G.is_dag:
                        G = sbn_release4.SBNGraph().from_path(filepath)
                        logging.warning(
                            "Scopeless DRG is cyclic; use the complete DRG instead. "
                        )
                elif args.scopeless:
                    G= sbn_scopeless4.SBNGraph().from_path(filepath)
                    if not G.is_dag:
                        G = sbn_release4.SBNGraph().from_path(filepath)
                        logging.warning(
                            "Scopeless DRG is cyclic; use the complete DRG instead. "
                        )
                elif args.gold:
                    G= sbn_release4.SBNGraph().from_path(filepath)
                    penman_string = G.to_penman_string()
                    output_path = Path(f"{filepath.parent}/{experiment}/{filepath.stem}.penman")
                    sbn_release5.to_penman(penman_string, output_path)
            elif args.version==5:
                if args.scopeless:
                    G = sbn_scopeless5.SBNGraph().from_path(filepath)
                    if not G.is_dag:
                        G = sbn_release5.SBNGraph().from_path(filepath)
                        logging.warning(
                            "Scopeless DRG is cyclic; use the complete DRG instead. "
                        )
                elif args.simplified:
                    G = sbn_simplified5.SBNGraph().from_path(filepath)
                    if not G.is_dag:
                        G = sbn_release5.SBNGraph().from_path(filepath)
                        logging.warning(
                            "Scopeless DRG is cyclic; use the complete DRG instead. "
                        )

                elif args.gold:
                    G = sbn_release5.SBNGraph().from_path(filepath)
                    penman_string = G.to_penman_string()
                    output_path = Path(f"{filepath.parent}/{experiment}/{filepath.stem}.penman")
                    sbn_release5.to_penman(penman_string, output_path)
            if args.png:
                G.to_png(f'{filepath.parent}/{experiment}/{filepath.stem}.png')


            if args.simplified or args.scopeless:
                edges_info = get_edge_info(G, pre_map)
                nodes_info, comment_node_pair, cleaned_comment_list = get_node_info(G, pre_map, raw_path)
                comment_list_reference = deepcopy(cleaned_comment_list)
                comment_node_pair = group_nodes_for_alignment(nodes_info, edges_info, comment_node_pair)
                alignment_list = alignment(comment_node_pair, cleaned_comment_list, nodes_info, edges_info)
                penman_string = G.to_penman_string()
                output_path = Path(f"{filepath.parent}/{experiment}/{filepath.stem}.penman")
                sbn_release5.to_penman(penman_string, output_path)
                triples = penman.decode(penman_string).triples
                output_path_penmaninfo = Path(f"{filepath.parent}/{experiment}/{filepath.stem}.penmaninfo")

                with open(output_path_penmaninfo, 'w') as penmaninfo:
                    for t in triples:
                        penmaninfo.write(f'{t}\n')
                    penmaninfo.write('\ntokenized sentence:')
                    tokenized_sent = ' '.join([x[0] for x in comment_list_reference])
                    penmaninfo.write(tokenized_sent)
                    penmaninfo.write('\n\n')
                    penmaninfo.write('\nalignment:')
                    json.dump(alignment_list, penmaninfo)
            # train_correct.write(
            #         f"{filepath.parent.parent.parent.name}/{filepath.parent.parent.name}/{filepath.parent.name}/{experiment}" + "\n")
            if f"{filepath.parent.parent.name}/{filepath.parent.name}" in train_split:
                train_correct.write(f"{filepath.parent.parent.parent.name}/{filepath.parent.parent.name}/{filepath.parent.name}/{experiment}"+"\n")
            elif f"{filepath.parent.parent.name}/{filepath.parent.name}" in dev_split:
                dev_correct.write(f"{filepath.parent.parent.parent.name}/{filepath.parent.parent.name}/{filepath.parent.name}/{experiment}"+"\n")
            elif f"{filepath.parent.parent.name}/{filepath.parent.name}" in test_split:
                test_correct.write(f"{filepath.parent.parent.parent.name}/{filepath.parent.parent.name}/{filepath.parent.name}/{experiment}" + "\n")
            elif f"{filepath.parent.parent.name}/{filepath.parent.name}" in other_split:
                other_correct.write(f"{filepath.parent.parent.parent.name}/{filepath.parent.parent.name}/{filepath.parent.name}/{experiment}" + "\n")

        except (SBNError, AlignmentError, KeyError, sbn_spec.SBNError, FileNotFoundError, RecursionError, penman.exceptions.DecodeError, KeyError) as e:
        # except:
            error += 1
            if args.log_error:
                print(error)
                print(f'error {filepath}')
                print(f'Error type: {type(e).__name__}')
                # print(e)
                # G.to_png(f'{filepath.parent}/{experiment}/{filepath.stem}.png')

            continue


generate_gold_split(train_path, gold_train, args.version)
generate_gold_split(dev_path, gold_dev, args.version)
generate_gold_split(other_path, gold_other, args.version)
generate_gold_split(test_path, gold_test, args.version)





