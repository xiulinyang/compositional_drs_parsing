import logging

from base import BaseGraph
from sbn import SBNGraph
from sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE

logger = logging.getLogger(__name__)

# Possible useful to extract Role mappings from:
#   - https://universaldependencies.org/u/feat/Degree.html for equality
#   - https://universaldependencies.org/u/feat/Person.html for speaker constant etc.
#   - https://universaldependencies.org/u/feat/Aspect.html for temporal relations, in addition to verb tense noted above
#   - https://universaldependencies.org/u/feat/Mood.html similar to Aspect and Tense


__all__ = ["GraphTransformer", "BoxRemover"]


class GraphTransformer:
    @staticmethod
    def transform(G: BaseGraph, **kwargs) -> BaseGraph:
        raise NotImplemented


class BoxRemover(GraphTransformer):
    @staticmethod
    def transform(G: SBNGraph, **kwargs) -> BaseGraph:
        edges_to_remove = {
            edge_id
            for edge_id, edge_data in G.edges.items()
            if edge_data["type"]
            in (SBN_EDGE_TYPE.BOX_CONNECT, SBN_EDGE_TYPE.BOX_BOX_CONNECT)
        }
        nodes_to_remove = {
            node_id
            for node_id, node_data in G.nodes.items()
            if node_data["type"] == SBN_NODE_TYPE.BOX
        }

        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(nodes_to_remove)

        return G
