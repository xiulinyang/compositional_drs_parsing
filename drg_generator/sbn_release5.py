from __future__ import annotations
import re
import logging
import collections
from collections import Counter
import os
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
# from helpers import pmb_generator
import networkx as nx
import penman
from tqdm import tqdm
from base import BaseEnum, BaseGraph
from graph_resolver import GraphResolver
from misc import ensure_ext
from penman.graph import Graph
from penman_model import pm_model
from sbn_spec_release5 import (
    SBN_EDGE_TYPE,
    SBN_NODE_TYPE,
    SBNError,
    AlignmentError,
    SBNSpec,
    split_comments,
    split_single,
    split_synset_id,
)


stopwords = ['the', 'a', 'to', 'in']
# load_model = spacy.load('en_core_web_sm', disable=['parser', 'ner'])



logger = logging.getLogger(__name__)

RESOLVER = GraphResolver()

__all__ = [
    "SBN_ID",
    "SBNGraph",
    "sbn_graphs_are_isomorphic",
]

# Node / edge ids, unique combination of type and index / count for the current
# document.
SBN_ID = Tuple[Union[SBN_NODE_TYPE, SBN_EDGE_TYPE], int]


class SBNSource(BaseEnum):
    # The SBNGraph is created from an SBN file that comes from the PMB directly
    PMB = "PMB"
    # The SBNGraph is created from GREW output
    GREW = "GREW"
    # The SBNGraph is created from a self generated SBN file
    INFERENCE = "INFERENCE"
    # The SBNGraph is created from a seq2seq generated SBN line
    SEQ2SEQ = "SEQ2SEQ"
    # We don't know the source or it is 'constructed' manually
    UNKNOWN = "UNKNOWN"


class SBNGraph(BaseGraph):
    def __init__(
            self,
            incoming_graph_data=None,
            source: SBNSource = SBNSource.UNKNOWN,
            **attr,
    ):
        super().__init__(incoming_graph_data, **attr)
        self.is_dag: bool = False
        self.is_possibly_ill_formed: bool = False
        self.source: SBNSource = source

    def from_path(self, path: PathLike) -> SBNGraph:
        """Construct a graph from the provided filepath."""
        return self.from_string(Path(path).read_text())

    def from_string(self, input_string: str, is_single_line=False) -> SBNGraph:
        """Construct a graph from a single SBN string."""
        # Determine if we're dealing with an SBN file with newlines (from the
        # PMB for instance) or without (from neural output).

        if is_single_line:
            input_string = split_single(input_string)
        lines = split_comments(input_string)

        if not lines:
            raise SBNError(
                "SBN doc appears to be empty, cannot read from string"
            )

        self.__init_type_indices()

        starting_box = self.create_node(
            SBN_NODE_TYPE.BOX, self._active_box_token,
            {"comment": "START0"}
        )

        nodes, edges = [starting_box], []
        max_wn_idx = len(lines) - 1
        #change1: collect important node information
        node_info = [x.split()[0] if len(x.split()[0].split('.')) == 3 else [x.split()[0], x.split()[1]] for x, _ in lines]

        node_only_info = [x.split() for x, _ in lines if len(x.split()[0].split('.'))==3]
        proposition_list = [(i, x[j+1]) for i, x in enumerate(node_only_info) for j, y in enumerate(x) if y=='Proposition']
        scope = split_list_using_separators(node_info, proposition_list)
        discourse_connective =0

        for sbn_line, comment in lines:
            tokens = sbn_line.split()

            tok_count = 0
            while len(tokens) > 0:
                # Try to 'consume' all tokens from left to right
                token: str = tokens.pop(0)

                # No need to check all tokens for this since only the first
                # might be a sense id.
                if tok_count == 0 and (
                        synset_match := SBNSpec.SYNSET_PATTERN.match(token)
                ):
                    synset_node = self.create_node(
                        SBN_NODE_TYPE.SYNSET,
                        token,
                        {"wn_lemma": synset_match.group(),
                         "comment": comment,
                         },
                    )
                    nodes.append(synset_node)

                elif token in SBNSpec.NEW_BOX_INDICATORS:
                    # In the entire dataset there are no indices for box
                    # references other than -1. Maybe they are needed later and
                    # the exception triggers if something different comes up.
                    if not tokens:
                        raise SBNError(
                            f"Missing box index in line: {sbn_line}"
                        )
                    # change2: special treatment of proposition
                    if token=='Proposition':
                        tokens.pop(0)
                        continue

                    box_index = tokens.pop(0)
                    current_box_id = self._current_box_id(box_index)
                    new_box = self.create_node(
                        SBN_NODE_TYPE.BOX, self._active_box_token,
                        {'token': token,
                         'comment': comment,
                         'index': box_index})

                    nodes.append(new_box)

                    if not proposition_list:
                        box_edge = self.create_edge(
                                current_box_id,
                                self._active_box_id,
                                SBN_EDGE_TYPE.BOX_BOX_CONNECT,
                                token,
                            )

                        edges.append(box_edge)
                    #change3: mark the discourse connective index
                    discourse_connective+=1
                elif (is_role := token in SBNSpec.ROLES) or (
                        token in SBNSpec.DRS_OPERATORS
                ):
                    if not tokens:
                        raise SBNError(
                            f"Missing target for '{token}' in line {sbn_line}"
                        )

                    target = tokens.pop(0)
                    edge_type = (
                        SBN_EDGE_TYPE.ROLE
                        if is_role
                        else SBN_EDGE_TYPE.DRS_OPERATOR
                    )

                    if index_match := SBNSpec.INDEX_PATTERN.match(target):
                        idx = self._try_parse_idx(index_match.group(0))
                        active_id = self._active_synset_id
                        target_idx = active_id[1] + idx
                        to_id = (active_id[0], target_idx)

                        if SBNSpec.MIN_SYNSET_IDX <= target_idx <= max_wn_idx:
                            role_edge = self.create_edge(
                                self._active_synset_id,
                                to_id,
                                edge_type,
                                token,
                            )

                            edges.append(role_edge)
                        else:
                            # A special case where a constant looks like an idx
                            # Example:
                            # pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn
                            # This is detected by checking if the provided
                            # index points at an 'impossible' line (synset) in
                            # the file.

                            # NOTE: we have seen that the neural parser does
                            # this very (too) frequently, resulting in arguably
                            # ill-formed graphs.
                            self.is_possibly_ill_formed = True

                            const_node = self.create_node(
                                SBN_NODE_TYPE.CONSTANT,
                                target,
                                {"comment": comment},
                            )
                            role_edge = self.create_edge(
                                self._active_synset_id,
                                const_node[0],
                                edge_type,
                                token,
                            )
                            nodes.append(const_node)
                            edges.append(role_edge)
                    elif SBNSpec.NAME_CONSTANT_PATTERN.match(target):
                        name_parts = [target]

                        # Some names contain whitspace and need to be
                        # reconstructed
                        while not target.endswith('"'):
                            target = tokens.pop(0)
                            name_parts.append(target)

                        # This is faster than constantly creating new strings
                        name = " ".join(name_parts)

                        name_node = self.create_node(
                            SBN_NODE_TYPE.CONSTANT,
                            name,
                            {"comment": comment},
                        )
                        role_edge = self.create_edge(
                            self._active_synset_id,
                            name_node[0],
                            SBN_EDGE_TYPE.ROLE,
                            token,
                        )

                        nodes.append(name_node)
                        edges.append(role_edge)
                    else:
                        const_node = self.create_node(
                            SBN_NODE_TYPE.CONSTANT,
                            target,
                            {"comment": comment},
                        )
                        role_edge = self.create_edge(
                            self._active_synset_id,
                            const_node[0],
                            SBN_EDGE_TYPE.ROLE,
                            token,
                        )

                        nodes.append(const_node)
                        edges.append(role_edge)
                else:
                    raise SBNError(
                        f"Invalid token found '{token}' in line: {sbn_line}"
                    )
                tok_count += 1
        #chagne4: add scope
        '''The new graph structure adds additional difficulty to graph conversion, particularly
        in the case of Proposition;
        my solution is that I treat the sbns that contain proposition differently from other sbns.
        when there is a proposition node, I will continue, and I will create a box until I meet
        the CONTINUATION edge.
        to be added'''

        for k,discourse_unit in enumerate(scope):
            if type(discourse_unit[0])!=list: #TODO: not sure why sometimes discourse_unit[0]!=list.
                discourse_unit =[discourse_unit]
                scope[k] = discourse_unit

            from_box_node = (SBN_NODE_TYPE.BOX, int(discourse_unit[0][-1]))
            for node in discourse_unit[1:]:
                to_node = (SBN_NODE_TYPE.SYNSET, node)
                box_edge = self.create_edge(from_box_node, to_node,
                                            SBN_EDGE_TYPE.BOX_CONNECT)
                edges.append(box_edge)
        if proposition_list:
            other_discourse_connectives = [x for x in scope if x[0][0] not in ['Proposition', 'ROOT']]
            for connective in other_discourse_connectives:
                from_index = connective[0][2]+int(connective[0][1])
                from_box = (SBN_NODE_TYPE.BOX, scope[from_index][0][2])
                to_box = (SBN_NODE_TYPE.BOX, connective[0][2])
                box_box_edge = self.create_edge(from_box, to_box,
                                                    SBN_EDGE_TYPE.BOX_BOX_CONNECT,
                                                    connective[0][0])
                edges.append(box_box_edge)
            prop_nodes = [x[0][1] for x in scope if x[0][0]=='Proposition']
            prop_boxes = [x[0][2] for x in scope if x[0][0]=='Proposition']
            for prop_node, prop_box in zip(prop_nodes, prop_boxes):
                to_box = (SBN_NODE_TYPE.BOX, prop_box)
                proposition_predicate = (SBN_NODE_TYPE.SYNSET, prop_node)
                proposition_edge = self.create_edge(proposition_predicate, to_box,
                                                    SBN_EDGE_TYPE.BOX_BOX_CONNECT,
                                                    'Proposition')
                edges.append(proposition_edge)
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        self._check_is_dag()
        return self

    def create_edge(
            self,
            from_node_id: SBN_ID,
            to_node_id: SBN_ID,
            type: SBN_EDGE_TYPE,
            token: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ):
        """Create an edge, if no token is provided, the id will be used."""
        edge_id = self._id_for_type(type)
        meta = meta or dict()
        return (
            from_node_id,
            to_node_id,
            {
                "_id": str(edge_id),
                "type": type,
                "type_idx": edge_id[1],
                "token": token or str(edge_id),
                **meta,
            },
        )

    def create_node(
            self,
            type: SBN_NODE_TYPE,
            token: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ):
        """Create a node, if no token is provided, the id will be used."""
        node_id = self._id_for_type(type)
        meta = meta or dict()
        return (
            node_id,
            {
                "_id": str(node_id),
                "type": type,
                "type_idx": node_id[1],
                "token": token or str(node_id),
                **meta,
            },
        )

    def to_sbn(self, path: PathLike, add_comments: bool = False) -> Path:
        """Writes the SBNGraph to a file in sbn format"""
        final_path = ensure_ext(path, ".sbn")
        final_path.write_text(self.to_sbn_string(add_comments))
        return final_path

    def to_sbn_string(self, add_comments: bool = False) -> str:
        """Creates a string in sbn format from the SBNGraph"""
        #TODO: release5.0.0 has box indices other than -1, and therefore it should be changed!
        result = []
        synset_idx_map: Dict[SBN_ID, int] = dict()
        line_idx = 0

        box_nodes = [
            node for node in self.nodes if node[0] == SBN_NODE_TYPE.BOX
        ]
        for box_node_id in box_nodes:
            box_box_connect_to_insert = None
            for edge_id in self.out_edges(box_node_id):
                _, to_node_id = edge_id
                to_node_type, _ = to_node_id

                edge_data = self.edges.get(edge_id)
                if edge_data["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
                    if box_box_connect_to_insert:
                        raise SBNError(
                            "Found box connected to multiple boxes, "
                            "is that possible?"
                        )
                    else:
                        box_box_connect_to_insert = edge_data["token"]

                if to_node_type in (
                        SBN_NODE_TYPE.SYNSET,
                        SBN_NODE_TYPE.CONSTANT,
                ):
                    if to_node_id in synset_idx_map:
                        raise SBNError(
                            "Ambiguous synset id found, should not be possible"
                        )

                    synset_idx_map[to_node_id] = line_idx
                    temp_line_result = [to_node_id]
                    for syn_edge_id in self.out_edges(to_node_id):
                        _, syn_to_id = syn_edge_id

                        syn_edge_data = self.edges.get(syn_edge_id)
                        if syn_edge_data["type"] not in (
                                SBN_EDGE_TYPE.ROLE,
                                SBN_EDGE_TYPE.DRS_OPERATOR,
                        ):
                            raise SBNError(
                                f"Invalid synset edge connect found: "
                                f"{syn_edge_data['type']}"
                            )

                        temp_line_result.append(syn_edge_data["token"])

                        syn_node_to_data = self.nodes.get(syn_to_id)
                        syn_node_to_type = syn_node_to_data["type"]
                        if syn_node_to_type == SBN_NODE_TYPE.SYNSET:
                            temp_line_result.append(syn_to_id)
                        elif syn_node_to_type == SBN_NODE_TYPE.CONSTANT:
                            temp_line_result.append(syn_node_to_data["token"])
                        else:
                            raise SBNError(
                                f"Invalid synset node connect found: "
                                f"{syn_node_to_type}"
                            )

                    result.append(temp_line_result)
                    line_idx += 1
                elif to_node_type == SBN_NODE_TYPE.BOX:
                    pass
                else:
                    raise SBNError(f"Invalid node id found: {to_node_id}")

            if box_box_connect_to_insert:
                result.append([box_box_connect_to_insert, "-1"])

        # Resolve the indices and the correct synset tokens and create the sbn
        # line strings for the final string
        final_result = []
        if add_comments:
            final_result.append(
                (
                    f"{SBNSpec.COMMENT_LINE} SBN source: {self.source.value}",
                    " ",
                )
            )
        current_syn_idx = 0
        for line in result:
            tmp_line = []
            comment_for_line = None

            for token_idx, token in enumerate(line):
                # There can never be an index at the first token of a line, so
                # always start at the second token.
                if token_idx == 0:
                    # It is a synset id that needs to be converted to a token
                    if token in synset_idx_map:
                        node_data = self.nodes.get(token)
                        tmp_line.append(node_data["token"])
                        comment_for_line = comment_for_line or (
                            node_data["comment"]
                            if "comment" in node_data
                            else None
                        )
                        current_syn_idx += 1
                    # It is a regular token
                    else:
                        tmp_line.append(token)
                # It is a synset which needs to be resolved to an index
                elif token in synset_idx_map:
                    target = synset_idx_map[token] - current_syn_idx + 1
                    # In the PMB dataset, an index of '0' is written as '+0',
                    # so do that here as well.
                    tmp_line.append(
                        f"+{target}" if target >= 0 else str(target)
                    )
                # It is a regular token
                else:
                    tmp_line.append(token)

            if add_comments and comment_for_line:
                tmp_line.append(f"{SBNSpec.COMMENT}{comment_for_line}")

            # This is a bit of trickery to vertically align synsets just as in
            # the PMB dataset.
            if len(tmp_line) == 1:
                final_result.append((tmp_line[0], " "))
            else:
                final_result.append((tmp_line[0], " ".join(tmp_line[1:])))

        # More formatting and alignment trickery.
        max_syn_len = max(len(s) for s, _ in final_result) + 1
        sbn_string = "\n".join(
            f"{synset: <{max_syn_len}}{rest}".rstrip(" ")
            for synset, rest in final_result
        )

        return sbn_string


    def _current_box_token(self, nodes):
        current_id = self._active_box_id
        box_node = [x for x in nodes if x[0]==current_id][0][1]['token']
        if box_node!='B-0':
            box_index = [x for x in nodes if x[0]==current_id][0][1]['index']
        else:
            box_index='0'
        return box_node, box_index
    def to_penman_string(
            self, strict: bool = True
    ) -> str:
        """
        Creates a string in Penman (AMR-like) format from the SBNGraph.

        The 'evaluate_sense; flag indicates if the sense number is included.
        If included, the evaluation indirectly also targets the task of word
        sense disambiguation, which might not be desirable. Example:

            (b0 / "box"
                :member (s0 / "synset"
                    :lemma "person"
                    :pos "n"
                    :sense "01")) # Would be excluded when False

        The 'strict' option indicates how to handle possibly ill-formed graphs.
        Especially when indices point at impossible synsets. Cyclic graphs are
        also ill-formed, but these are not even allowed to be exported to
        Penman.

        FIXME: the DRS/SBN constants technically don't need a variable. As long
        as this is consistent between the gold and generated data, it's not a
        problem.
        """
        G = deepcopy(self)

        prefix_map = {
            SBN_NODE_TYPE.BOX: ["b", 0],
            SBN_NODE_TYPE.CONSTANT: ["c", 0],
            SBN_NODE_TYPE.SYNSET: ["s", 0],
        }

        for node_id, node_data in G.nodes.items():

            pre, count = prefix_map[node_data["type"]]
            prefix_map[node_data["type"]][1] += 1  # type: ignore
            G.nodes[node_id]["var_id"] = f"{pre}{count}"

            # A box is always an instance of the same type (or concept), the
            # specification of what that type does is shown by the
            # box-box-connection, such as NEGATION or EXPLANATION.
            if node_data["type"] == SBN_NODE_TYPE.BOX:
                G.nodes[node_id]["token"] = "box"

        for edge in G.edges:
            # Add a proper token to the box connectors
            if G.edges[edge]["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
                G.edges[edge]["token"] = "member"

        def __to_penman_str(S: SBNGraph, current_n, visited, out_str, tabs):
            node_data = S.nodes[current_n]
            var_id = node_data["var_id"]
            if var_id in visited:
                out_str += var_id
                return out_str

            indents = tabs * "\t"
            node_tok = node_data["token"]

            if strict:
                if node_data["type"] == SBN_NODE_TYPE.SYNSET:
                    if not (components := split_synset_id(node_tok)):
                        raise SBNError(f"Cannot split synset id: {node_tok}")
                    lemma, pos, sense = [self.quote(i) for i in components]
                    ### changed part
                    wordnet = lemma.strip('"') + '.' + pos.strip('"') + '.' + sense.strip('"')
                    out_str += f'({var_id} / {wordnet}'
                elif var_id[0] != "c":
                    out_str += f"({var_id} / {node_tok}"
                else:
                    out_str += f"{self.quote(node_tok)}"
            else:  # if strict == False
                if node_data["type"] == SBN_NODE_TYPE.SYNSET:
                    if not (components := split_synset_id(node_tok)):
                        raise SBNError(f"Cannot split synset id: {node_tok}")
                    lemma, pos, sense = [self.quote(i) for i in components]
                    out_str += f'({var_id} / {self.quote("synset")}'
                    out_str += f"\n{indents}:lemma {lemma}"
                    out_str += f"\n{indents}:pos {pos}"
                    # out_str += f"\n{indents}:sense {sense}"
                    """this part should be checked if same as Wessel's evaluation"""
                else:
                    out_str += f"({var_id} / {self.quote(node_tok)}"

            if S.out_degree(current_n) > 0:
                for edge_id in S.edges(current_n):
                    edge_name = S.edges[edge_id]["token"]
                    if edge_name in SBNSpec.INVERTIBLE_ROLES:
                        # SMATCH can invert edges that end in '-of'.
                        # This means that,
                        #   A -[AttributeOf]-> B
                        #   B -[Attribute]-> A
                        # are treated the same, but they need to be in the
                        # right notation for this to work.
                        edge_name = edge_name.replace("Of", "-of")

                    _, child_node = edge_id
                    out_str += f"\n{indents}:{edge_name} "
                    out_str = __to_penman_str(
                        S, child_node, visited, out_str, tabs + 1
                    )
            if var_id[0] == "c":
                visited.add(var_id)
            else:
                out_str += ")"
                visited.add(var_id)
            return out_str

        # Assume there always is the starting box to serve as the "root"
        starting_node = (SBN_NODE_TYPE.BOX, 0)
        final_result = __to_penman_str(G, starting_node, set(), "", 1)

        return final_result

    def __init_type_indices(self):
        self.type_indices = {
            SBN_NODE_TYPE.SYNSET: 0,
            SBN_NODE_TYPE.CONSTANT: 0,
            SBN_NODE_TYPE.BOX: 0,
            SBN_EDGE_TYPE.ROLE: 0,
            SBN_EDGE_TYPE.DRS_OPERATOR: 0,
            SBN_EDGE_TYPE.BOX_CONNECT: 0,
            SBN_EDGE_TYPE.BOX_BOX_CONNECT: 0,
        }

    def _id_for_type(
            self, type: Union[SBN_EDGE_TYPE, SBN_NODE_TYPE]
    ) -> SBN_ID:
        _id = (type, self.type_indices[type])
        self.type_indices[type] += 1
        return _id

    def _check_is_dag(self) -> bool:
        self.is_dag = nx.is_directed_acyclic_graph(self)
        if not self.is_dag:
            logger.warning(
                "Initialized cyclic SBN graph, this will work for most tasks, "
                "but can cause problems later on when exporting to Penman for "
                "instance."
            )
            raise SBNError("The graph is cyclic")
        return self.is_dag

    @staticmethod
    def _try_parse_idx(possible_idx: str) -> int:
        """Try to parse a possible index, raises an SBNError if this fails."""
        try:
            return int(possible_idx)
        except ValueError:
            raise SBNError(f"Invalid index '{possible_idx}' found.")

    @staticmethod
    def quote(in_str: str) -> str:
        """Consistently quote a string with double quotes"""
        if in_str.startswith('"') and in_str.endswith('"'):
            return in_str

        if in_str.startswith("'") and in_str.endswith("'"):
            return f'"{in_str[1:-1]}"'

        return f'"{in_str}"'

    @property
    def _active_synset_id(self) -> SBN_ID:
        return (
            SBN_NODE_TYPE.SYNSET,
            self.type_indices[SBN_NODE_TYPE.SYNSET] - 1)

    def _active_node_synset_id(self, target_node, reference) -> SBN_ID:

        return (SBN_NODE_TYPE.SYNSET,
                reference.index(target_node))

    @property
    def _active_box_id(self) -> SBN_ID:
        return (SBN_NODE_TYPE.BOX,
                self.type_indices[SBN_NODE_TYPE.BOX] - 1)


    def _current_box_id(self, box_index:str) -> SBN_ID:
        return (SBN_NODE_TYPE.BOX, self.type_indices[SBN_NODE_TYPE.BOX] + int(box_index))

    def _prev_box_id(self, offset: int) -> SBN_ID:
        n = self.type_indices[SBN_NODE_TYPE.BOX]
        return (
            SBN_NODE_TYPE.BOX,
            max(0, min(n, n - offset)),  # Clamp so we always have a valid box
        )

    @property
    def _active_box_token(self) -> str:
        return f"B-{self.type_indices[SBN_NODE_TYPE.BOX]}"


    @staticmethod
    def _node_label(node_data) -> str:
        return node_data["token"]
        # return "\n".join(f"{k}={v}" for k, v in node_data.items())

    @staticmethod
    def _edge_label(edge_data) -> str:
        return edge_data["token"]
        # return "\n".join(f"{k}={v}" for k, v in edge_data.items())

    @property
    def type_style_mapping(self):
        """Style per node type to use in dot export"""
        return {
            SBN_NODE_TYPE.SYNSET: {},
            SBN_NODE_TYPE.CONSTANT: {"shape": "none"},
            SBN_NODE_TYPE.BOX: {"shape": "box", "label": ""},
            SBN_EDGE_TYPE.ROLE: {},
            SBN_EDGE_TYPE.DRS_OPERATOR: {},
            SBN_EDGE_TYPE.BOX_CONNECT: {"style": "dotted", "label": ""},
            SBN_EDGE_TYPE.BOX_BOX_CONNECT: {},
        }

def to_penman(
    penman_info, path: PathLike
) -> PathLike:
    """
    Writes the SBNGraph to a file in Penman (AMR-like) format.

    See `to_penman_string` for an explanation of `strict`.
    """
    final_path = ensure_ext(path, ".penman")
    final_path.write_text(penman_info)
    return final_path
def split_list_using_separators(main_list, proposition_list):
    result = []
    current_sublist = []
    # print(proposition_list)
    c =0
    box_c =0
    for element in main_list:
        if len(element) ==2:
            box_c += 1
            if current_sublist:
                result.append(current_sublist)
            else:
                result.append([])
            element.append(box_c)

            current_sublist = [element]  # Clear the current sublist to start building a new one
        else:
            current_sublist.append(c)
            c+=1

    # Append the last sublist, if any
    if current_sublist:
        result.append((current_sublist))

    if proposition_list:
        for id, index in proposition_list:
            #id: the node id where the proposition lies
            #index: the index after proposition
            context_id = result.index([x for x in result if id in x][0])

            if int(index)>0:
                # if proposition >n
                assert result[context_id+int(index)][0][0] =='CONTINUATION' and result[context_id+int(index)][0][1] =='-0'
                result[context_id+int(index)][0] = ['Proposition', id, result[context_id+int(index)][0][2]]
            elif int(index)<0:
                # if proposition <n, move the CONTINUATION BOX to the target discourse unit with a new name 'Proposition'
                # id is the node id of the predicate

                assert result[context_id][0][0] =='CONTINUATION' and result[context_id][0][1]=='-0'
                useless_continuation = result[context_id].pop(0)
                if type(result[context_id+int(index)][0])==int:
                    # if the argument does not have any other discourse connective
                    result[context_id + int(index)].insert(0, ['Proposition', id, useless_continuation[-1]])
                else:
                    #otherwise
                    result.insert(context_id+int(index), ['Proposition', id, useless_continuation[-1]])

            else:
                raise SBNError('the graph might not be right.')
    for i, unit in enumerate(result):
        if i==0 and len(unit)==0:
            unit.insert(0, ['ROOT', '-0', 0])
        if len(unit)>0 and type(unit[0])==int:
            unit.insert(0, ['ROOT', '-0', 0])
    return result
def sbn_graphs_are_isomorphic(A: SBNGraph, B: SBNGraph) -> bool:
    """
    Checks if two SBNGraphs are isomorphic this is based on node and edge
    ids as well as the 'token' meta data per node and edge
    """

    # Type and count are already compared implicitly in the id comparison that
    # is done in the 'is_isomorphic' function. The tokens are important to
    # compare since some constants (names, dates etc.) need to be reconstructed
    # properly with their quotes in order to be valid.
    def node_cmp(node_a, node_b) -> bool:
        return node_a["token"] == node_b["token"]

    def edge_cmp(edge_a, edge_b) -> bool:
        return edge_a["token"] == edge_b["token"]

    return nx.is_isomorphic(A, B, node_cmp, edge_cmp)