from collections import Counter
from copy import copy
from typing import Any, Dict, List, Tuple

from config import Config
from sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNError, SBNSpec
from ud_spec import (
    GENDER_SYNSET_MAPPING,
    TIME_EDGE_MAPPING,
    UPOS_WN_POS_MAPPING,
    UDSpecBasic,
)

__all__ = [
    "GraphResolver",
]


class GraphResolver:
    """
    The NodeResolver processes output from GREW and resolves node/edge types,
    labels, connections and more.
    """

    RESOLVE_TIME_EDGE = "TIMERELATION"
    RESOLVE_NONE_EDGE = "NONE"
    RESOLVE_GENDER_NODE = "GENDER"

    # These are the protected fields in the node and edge data that need
    # special care in certain places, such as when merging SBNGraphs.
    PROTECTED_FIELDS = ["_id", "type", "type_idx", "token"]

    def __init__(
        self,
        language: Config.SUPPORTED_LANGUAGES = Config.SUPPORTED_LANGUAGES.EN,
    ) -> None:
        self.edge_mappings = Config.get_edge_mappings(language)
        self.lemma_sense_lookup = Config.get_lemma_sense(language)
        self.lemma_pos_sense_lookup = Config.get_lemma_pos_sense(language)
        # TODO: use flag if we want to use this
        # self.edge_clf_pipeline = Config.get_edge_clf(language)
        self.edge_clf_pipeline = None

    def node_token_type(
        self, node_data: Dict[str, str]
    ) -> Tuple[SBN_NODE_TYPE, str, Dict[str, str]]:
        if not (token_to_resolve := node_data.get("token", None)):
            raise SBNError(
                f"All nodes need the 'token' features. Node data: {node_data} "
                "(Likely a UD parsing error with wrong sentence boundaries)"
            )

        node_data = GraphResolver.filter_item_data(node_data)
        node_token = copy(token_to_resolve)

        gender = (
            self.parse_gender(node_data["Gender"])[0]
            if "Gender" in node_data
            else None
        )

        # A new box has been added in the grew step
        if token_to_resolve in SBNSpec.NEW_BOX_INDICATORS:
            node_type = SBN_NODE_TYPE.BOX
        # The synset has been added in the grew rewriting step
        elif SBNSpec.SYNSET_PATTERN.match(token_to_resolve):
            node_type = SBN_NODE_TYPE.SYNSET
        elif token_to_resolve == self.RESOLVE_GENDER_NODE:
            node_type = SBN_NODE_TYPE.SYNSET
            node_token = GENDER_SYNSET_MAPPING.get(
                gender, Config.DEFAULT_GENDER
            )
        # Otherwise try to format the token as a synset. This assumes
        # unwanted nodes (DET, PUNCT, AUX) are already removed.
        elif upos := node_data.get("upos", None):
            # TODO: some POS tags indicate constants (NUM, PROPN, etc)
            # Maybe fix that here as well.
            node_type = SBN_NODE_TYPE.SYNSET
            wn_pos = UPOS_WN_POS_MAPPING[upos]
            lemma = token_to_resolve
            lemma_pos = f"{lemma}.{wn_pos}"

            if synset := self.lemma_pos_sense_lookup.get(lemma_pos, None):
                node_token = synset
            elif synset := self.lemma_sense_lookup.get(lemma, None):
                node_token = synset
            elif upos == UDSpecBasic.POS.PROPN:
                # TODO: this would be the place to get info per node from a
                # NER system or mark them for later processing with a NER
                # system. Can also work for dates (NUM), countries, etc.

                # Not resolving the gender here and always using "female.n.02"
                # scores higher, but does not really make sense. This is
                # more fair and logical, but scores slightly lower.
                node_token = GENDER_SYNSET_MAPPING.get(
                    gender, Config.DEFAULT_GENDER
                )
                # node_token = "female.n.02"  # most common in training data
            else:
                # TODO: it would be nice to detect if a given synset is actually
                # present in WordNet. If not, we could try to split it and look
                # up possible compounds or just fall back to entity.n.01 or
                # something like that.
                node_token = f"{lemma}.{wn_pos}.01"
        else:
            # The default type is constant.
            node_type = SBN_NODE_TYPE.CONSTANT

        return node_type, node_token, node_data

    def edge_token_type(
        self, edge_name, nodes, from_id, to_id
    ) -> Tuple[SBN_EDGE_TYPE, str, Dict[str, str]]:
        edge_data = self.parse_edge_name(edge_name)
        if not (token_to_resolve := edge_data.get("token", None)):
            raise SBNError(
                f"All edges need the 'token' features. Edge data: {edge_data}"
            )

        edge_data = GraphResolver.filter_item_data(edge_data)
        edge_token = copy(token_to_resolve)
        edge_type = None
        deprel = edge_data.get("deprel", None)

        if token_to_resolve in SBNSpec.ROLES:
            edge_type = SBN_EDGE_TYPE.ROLE
        elif token_to_resolve in SBNSpec.DRS_OPERATORS:
            edge_type = SBN_EDGE_TYPE.DRS_OPERATOR
        elif token_to_resolve in SBNSpec.NEW_BOX_INDICATORS:
            if (
                from_id[0] == SBN_NODE_TYPE.BOX
                and to_id[0] == SBN_NODE_TYPE.BOX
            ):
                edge_type = SBN_EDGE_TYPE.BOX_BOX_CONNECT
            else:
                edge_type = SBN_EDGE_TYPE.BOX_CONNECT
        elif token_to_resolve == self.RESOLVE_TIME_EDGE:
            edge_type = SBN_EDGE_TYPE.ROLE
            # Not the nicest solution, but we need to figure out the
            # tense, which is a bit of a pain on the grew side.
            tenses = [
                n_data["Tense"]
                for _, n_data in nodes.items()
                if "Tense" in n_data
            ]
            if len(tenses) > 0:
                counts = Counter(tenses).most_common(1)
                edge_token = TIME_EDGE_MAPPING[counts[0][0]]
            else:
                edge_token = Config.DEFAULT_TIME_ROLE
        elif token_to_resolve == self.RESOLVE_NONE_EDGE:
            from_upos = nodes[from_id].get("upos", None)
            to_upos = nodes[to_id].get("upos", None)
            key_components = [from_upos, deprel, to_upos]

            if all(key_components):
                key = "-".join(key_components)
                if key in self.edge_mappings:
                    edge_token = self.edge_mappings[key]

                    # TODO: This type info should probably be included in
                    # the mappings.
                    if edge_token in SBNSpec.ROLES:
                        edge_type = SBN_EDGE_TYPE.ROLE
                    elif edge_token in SBNSpec.DRS_OPERATORS:
                        edge_type = SBN_EDGE_TYPE.DRS_OPERATOR

        if not edge_type:
            # The default role and type
            edge_type = SBN_EDGE_TYPE.ROLE
            edge_token = Config.DEFAULT_ROLE

        return edge_type, edge_token, edge_data

    def predict_edge(self, deprel, from_node_data, to_node_data) -> str:
        if not self.edge_clf_pipeline:
            raise SBNError("Edge clf is not enabled")

        feature_vec = self.encode(deprel, from_node_data, to_node_data)
        label = self.edge_clf_pipeline.predict([feature_vec])[0]
        return label

    @staticmethod
    def parse_edge_name(edge_name) -> Dict[str, str]:
        # Grew encodes edge data in a similar way Feats are encoded in UD Conll
        # parses, so a single string that has to be split up into components.
        edge_data = {
            key: value
            for key, value in [
                item.split("=") for item in edge_name.split(",")
            ]
        }
        # Grew encodes deprels in a peculiar way, reconstruct it here.
        deprel_comp = [
            edge_data[deprel_component]
            for deprel_component in ["1", "2"]
            if deprel_component in edge_data
        ]
        edge_data.pop("1", None)
        edge_data.pop("2", None)
        deprel = ":".join(deprel_comp) if deprel_comp else None
        edge_data["deprel"] = deprel

        return edge_data

    @staticmethod
    def parse_gender(gender: str):
        """
        The 'Gender' feat can have multiple values.

        TODO: Maybe expand this to parse feats in general.
        """
        return gender.split(",")

    @staticmethod
    def filter_item_data(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make sure protected fields from nodes or edges are not overwritten
        by renaming fields from the dict.
        """
        for key in GraphResolver.PROTECTED_FIELDS:
            if key in item:
                item[f"meta_{key}"] = item.pop(key)
        return item

    @staticmethod
    def encode(
        deprel, from_node_data: Dict[str, Any], to_node_data: Dict[str, Any]
    ) -> List[Any]:
        feature_cols = [
            "deprel",
            "Case",
            "Degree",
            "Gender",
            "Mood",
            "Number",
            "NumType",
            "Person",
            "Poss",
            "PronType",
            "Tense",
            "upos",
            "VerbForm",
            "Voice",
            "xpos",
        ]

        features = [
            deprel,
            *[from_node_data.get(k, None) for k in feature_cols],
            *[to_node_data.get(k, None) for k in feature_cols],
        ]

        return features
