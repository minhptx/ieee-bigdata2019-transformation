import collections
from typing import Dict, Tuple, List

import defaultlist
import numpy as np
from sklearn.model_selection import KFold

from datafc.datasource.column import values_jaccard, text_jaccard, histogram_nmi
from datafc.datasource.column.source import Column
from datafc.ml import MultiBinary
from datafc.syntactic.graphbased import Graph, Edge
from datafc.syntactic.graphbased.model import SyntacticModel
from datafc.transformation.operators import Operation


class GraphTransformationModel:
    def __init__(self, original_syntactic: SyntacticModel, target_syntactic: SyntacticModel):
        self.scoring_model: MultiBinary = MultiBinary[Column](GraphTransformationModel.compute_sim)
        self.original_syntactic: SyntacticModel = original_syntactic
        self.target_syntactic = target_syntactic

    @staticmethod
    def compute_sim(col1: Column, col2: Column) -> List[float]:
        return [values_jaccard(col1, col2), text_jaccard(col1, col2), histogram_nmi(col1, col2)]

    def fit(self):
        graphs = self.original_syntactic.graphs + self.target_syntactic.graphs
        label_cols: Dict[str, Column] = {}
        for graph in graphs:
            for edge in graph.get_all_edges():
                num_splits = 5
                if len(edge.values) <= 5:
                    num_splits = len(edge.values)
                kf = KFold(n_splits=num_splits)
                for train_indices, test_indices in kf.split(edge.values):
                    train_values = []
                    for index in train_indices:
                        train_values.append(edge.values[index])
                    column = Column(str(edge), str(edge), train_values)
                    label_cols[str(edge)] = column
        self.scoring_model.train(label_cols)

    def transform(self) -> Tuple[List[str], List[str]]:
        original_values: List[str] = []
        transformed_values: List[str] = []

        graph_to_transformation = self.generate_program()
        for graph in self.original_syntactic.graphs:
            sub_original_values: List[str] = defaultlist.defaultlist(lambda: "")
            sub_transformed_values: List[str] = defaultlist.defaultlist(lambda: "")
            operations, score = graph_to_transformation[graph]

            for operation in operations:
                transformed_sub_values = operation.transform()
                # print(operation, operation.score(self.scoring_model))
                # print(len(operation.original_values), len(transformed_sub_values), len(operation.target_values))
                assert (len(operation.original_token) == len(transformed_sub_values))
                print("Chosen", operation, operation.score(self.scoring_model), transformed_sub_values[:5],
                      operation.original_token[:5], operation.target_values[:5])
                for index in range(len(operation.original_token)):
                    sub_original_values[index] += operation.original_token[index]
                    sub_transformed_values[index] += transformed_sub_values[index]

            for index in range(len(graph.values)):
                sub_original_values[index] = graph.values[index]
            original_values.extend(sub_original_values)
            transformed_values.extend(sub_transformed_values)

        return original_values, transformed_values

    def generate_program(self) -> Dict[Graph, Tuple[List[Operation], float]]:
        graph_to_transformation: Dict[Graph, Tuple[List[Operation], float]] = {}
        for original_graph in self.original_syntactic.graphs:
            best_score: float = 0
            best_operations: List[Operation] = []
            for target_graph in self.target_syntactic.graphs:
                operations, score = self.best_transform_between_graph(original_graph, target_graph)
                if score > best_score:
                    best_operations = operations
                    best_score = score

            graph_to_transformation[original_graph] = (best_operations, best_score)
        return graph_to_transformation

    # TODO top k transformations

    def best_transform_between_graph(self, original_graph: Graph, target_graph: Graph) \
            -> Tuple[List[Operation], float]:
        edge_pair_to_score: Dict[Edge, List[Tuple[Operation, float]]] = collections.defaultdict(list)
        for target_edge in target_graph.get_all_edges():
            for original_edge in original_graph.get_all_edges():
                edge_pair_to_score[target_edge].extend([tup for tup in Operation.find_top_k_transformations(
                    original_edge.values, target_edge.values,
                    self.scoring_model).items()])
            for op, score in edge_pair_to_score[target_edge]:
                print(target_edge.values[:3], op.original_token[:3], op, op.score(self.scoring_model))

        target_paths = target_graph.to_paths()
        path_index_to_score: List[float] = defaultlist.defaultlist(lambda: 0)
        path_index_to_operations: Dict[int, List[Operation]] = collections.defaultdict(list)

        for idx, path in enumerate(target_paths):
            for i in range(len(path) - 1):
                edge = target_graph.nested_nodes_to_edge[path[i]][path[i + 1]]
                operation, score = max(edge_pair_to_score[edge], key=lambda x: x[1])
                path_index_to_score[idx] += score
                path_index_to_operations[idx].append(operation)

        best_path_index = np.argmax(path_index_to_score)[0]
        return path_index_to_operations[best_path_index], np.max(path_index_to_score)


