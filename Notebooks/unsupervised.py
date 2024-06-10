#! /usr/bin/env python3

"""
This file contains all the methods for unsupervised link prediction (mostly based on graph topology)
"""

from itertools import tee
import itertools
import math
import os
import random
import numpy as np
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import Ml_100k
from sklearn import metrics

class Metrics:
    def __init__(self, graph, save_folder=None) -> None:
        self.graph = graph
        self.data = {}
        self.normalized = False
        if save_folder is None:
            save_folder = os.path.join(os.getcwd(), "similarities")
        self.save_folder = save_folder
        self.load(self.save_folder)

    # TODO: Load/save does not care about which graph it saves/load. Graph Id need to be added.

    def load(self, folder):
        if not os.path.isdir(folder):
            return
        npy_files = [
            f
            for f in os.listdir(folder)
            if f[-4:] == ".npy" and os.path.isfile(os.path.join(folder, f))
        ]
        for file in npy_files:
            self.data[file[:-4]] = np.load(
                os.path.join(folder, file), allow_pickle=True
            )

    def reset(self):
        self.data = {}

    def save(self, save_folder=None, normalized_policy="forbid"):
        """
        One should be weary of saving it's similarities values with consistency regarding data normalization.
        It is recommended to save/load the similarities before normalizing them.
        """
        if self.normalized and normalized_policy == "forbid":
            return  # Actually forbid to save normalized data to evade inconsistencies
        if save_folder is not None:
            self.save_folder = save_folder
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        for metric, data in self.data.items():
            np.save(os.path.join(self.save_folder, f"{metric}.npy"), data)

    def exists(self, user, movie, metric):
        if not metric in self.data:
            self.data[metric] = np.full(
                shape=(self.graph.num_users, self.graph.num_movies),
                fill_value=None,
            )

    def add(self, user, movie, metric, value):
        self.exists(user, movie, metric)
        self.data[metric][user, movie - self.graph.num_users] = value

    def normalize(self):
        """
        Normalize all the similarity metrics.
        !! Beware that you won't be able to save the data after normalizing it !!
        """
        for mat in self.data.values():
            mat[mat is None] = 0
            mat.astype("float64", copy=False)
            mat -= mat.min()
            mat /= mat.max()

    def __str__(self) -> str:
        res = ""
        for metric, mat in self.data.items():
            res += f"{metric}:\n{str(mat)}"
        return res + "\n"

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data.get(index, None)
        if not index[0] in self.data:
            return None
        if len(index) == 2:  # Honestly I don't think it is wise to call this case.
            metric, user, movie = index
            return self.data[metric][user, :]
        metric, user, movie = index
        return self.data[metric][user, movie - self.graph.num_users]

    def plot(self, xscale="linear", yscale="linear"):
        values = {
            metric: -np.sort(-mat.reshape(-1)) for metric, mat in self.data.items()
        }

        for mat in values.values():
            plt.plot(mat)
        plt.legend([metric for metric in values.keys()])
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlabel("edges")
        plt.ylabel("normalized link prediction value")
        plt.show()


class Graph(object):

    def __init__(
        self, data, data_type="biadjacency", evaluation=0.2, save_folder=None
    ) -> None:
        match data_type:
            case "biadjacency":
                self.num_users = data.shape[0]
                self.num_movies = data.shape[1]
                self.G_train = nx.bipartite.from_biadjacency_matrix(
                    sparse.coo_matrix(data)
                )
                self.G_orig = self.G_train.copy()
                self.G_test = None
                self.test_set = []
            case "edge-list":
                self.num_users = len(np.unique(data[0]))
                self.num_movies = len(np.unique(data[1]))
                tmp = nx.Graph()
        print(f"nb users: {self.num_users}, nb movies: {self.num_movies}")
        self.evaluation = 0 < evaluation < 1
        if self.evaluation:
            self.split(test_sample_size=evaluation)
        self.similarities = Metrics(self, save_folder)

    def split(self, test_sample_size):
        non_edges = list(nx.non_edges(self.G_train))
        test_edges = random.sample(
            list(self.G_train.edges()),
            int(self.G_train.number_of_edges() * test_sample_size),
        )
        self.G_test = self.G_train.copy()
        self.G_train.remove_edges_from(test_edges)
        self.test_set = test_edges + non_edges[:len(test_edges)]

    def compute_similarity(self, similarity, edge_subset=None):
        """Wrapper for all similarities, make sure not to recompute any value."""

        if edge_subset is None:
            users = [n for n, v in self.G_train.nodes(data=True) if v["bipartite"] == 0]
            movies = [
                n for n, v in self.G_train.nodes(data=True) if v["bipartite"] == 1
            ]
            edge_subset = list(itertools.product(users, movies))

        match similarity:
            case "common-neighbors":
                pretty_similarity_name = "common neighbors"
            case "adamic-adar":
                pretty_similarity_name = "Adamic-Adar"
            case "jaccard":
                pretty_similarity_name = "Jaccard's coefficients"
            case "preferential-attachment":
                pretty_similarity_name = "preferential attachment"
            case _:
                raise ValueError(f"Unknown similarity metric requested: {similarity}")

        for user, movie in tqdm(
            edge_subset,
            f"Computing {pretty_similarity_name}",
            unit=" potential edge(s)",
            unit_scale=True,
        ):
            if self.similarities[similarity, user, movie] is not None:
                continue
            match similarity:
                case "common-neighbors" | "adamic-adar" | "jaccard":
                    value = self.common_neighbors(user, movie, similarity)
                case "preferential-attachment":
                    value = self.preferential_attachment(user, movie)
            self.similarities.add(user, movie, similarity, value)

    def common_neighbors(self, user, movie, variation="common-neighbors"):
        """
        Instead of computing the intersection of the neighborhoods that would be 0 because the graph is bipartite,
        we compute the number of edges that connects these neighborhoods.

        For the Adamic-Adar variation, the weight for each edge (u,v) is 1/log(d(u)+d(v)) where d(u) is the degree of node u.

        For the Jaccard Variation, we divide by the number of edges link to at least one neighborhood.
        """
        res = 0
        if variation == "jaccard":
            len_union = 0
            for node in nx.neighbors(self.G_train, user):
                len_union += self.G_train.degree(node)
            for node in nx.neighbors(self.G_train, movie):
                len_union += self.G_train.degree(node)
        for m in nx.neighbors(self.G_train, user):
            for u in nx.neighbors(self.G_train, movie):
                if self.G_train.has_edge(u, m) or self.G_train.has_edge(m, u):
                    match variation:
                        case "common-neighbors":
                            res += 1
                        case "adamic-adar":
                            res += 1 / math.log(
                                self.G_train.degree(u) + self.G_train.degree(m)
                            )
                        case "jaccard":
                            res += 1
                            len_union -= 1
        if variation == "jaccard":
            if len_union == 0:
                assert res == 0
                return 0
            return res / len_union
        return res

    def preferential_attachment(self, user, movie):
        return self.G_train.degree(user) * self.G_train.degree(movie)

    def mean_average_precision(self, method, threshold):
        """
        Precision measure taken from https://towardsdatascience.com/link-prediction-in-bipartite-graph-ad766e47d75c.
        I don't fully understand it yet...
        """
        precisions = []
        for node in tqdm(self.G_test.nodes()):
            if node < self.num_users:
                predicted_edges = self.similarities[method][node, :]
            else:
                predicted_edges = self.similarities[method][:, node - self.num_users]
            ranked_predicted_edges = np.argsort(-predicted_edges)
            ranked_predictions = filter(
                lambda x: predicted_edges[x] > threshold, ranked_predicted_edges
            )
            prediction = [x for x in ranked_predictions]
            gt = set(self.G_test.neighbors(node))
            rel = np.array([x in gt for x in prediction])
            P = np.array(
                [sum(rel[: i + 1]) / len(rel[: i + 1]) for i in range(len(prediction))]
            )

            precisions.append((rel @ P) / len(gt))
        return np.mean(precisions)

    def prediction(self, method, threshold):
        """
        Returns the predictions
        """
        for node in tqdm(self.test_set):
            if node < self.num_users:
                predicted_edges = self.similarities[method][node, :]
            else:
                predicted_edges = self.similarities[method][:, node - self.num_users]
            ranked_predicted_edges = np.argsort(-predicted_edges)
            ranked_predictions = filter(
                lambda x: predicted_edges[x] > threshold, ranked_predicted_edges
            )
            prediction = [x for x in ranked_predictions]
        return prediction

    def labels(self, method):
        """
        Returns the predictions and ground truth labels.
        """
        y_pred = []
        y_true = []

        for user in tqdm(range(self.num_users)):
            predicted_edges = self.similarities[method][user, :]
            for movie in range(self.num_movies):
                y_pred.append(predicted_edges[movie])
                y_true.append(self.G_test.has_edge(user, movie + self.num_users))

        return y_pred, y_true

    def pred_true(self, method, threshold, provide_indices=False):
        """
        Returns the predictions and ground truth labels.
        """
        y_pred = []
        y_true = []
        indices = []
        for node_1, node_2 in tqdm(self.test_set):
            if node_1 < self.num_users:
                user, movie = node_1, node_2 - self.num_users
            else:
                user, movie = node_2, node_1 - self.num_users
            prediction = self.similarities[method][user, movie]
            # ranked_predicted_edges = np.argsort(-predicted_edge)
            # for edge in ranked_predicted_edges:
            if prediction > threshold:
                indices.append((user, movie))
                y_pred.append(prediction)
                y_true.append(self.G_test.has_edge(node_1, node_2))
        if provide_indices:
            return y_pred, y_true, indices
        return y_pred, y_true

    def plot_ROC(self, method):
        # Get predictions and ground truth labels
        y_pred, y_true = self.pred_true(method, 0)
        Binarypred = [1 if score > 0.5 else 0 for score in y_pred]
        ground_truth = list(map(int, y_true))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout()

        # Precision-Recall curve
        precision, recall, _ = metrics.precision_recall_curve(ground_truth, y_pred)
        ax1.plot(recall, precision)
        ax1.set_xlabel("Precision")
        ax1.set_ylabel("Recall")
        ax1.set_title("Precision-Recall Curve")

        # Receiver Operating Characteristic (ROC) curve
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        ax2.plot(fpr, tpr)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title(f"ROC Curve, AUC = {metrics.roc_auc_score(ground_truth, y_pred):.2f}")

        plt.show()

    def plot(self):
        pos = nx.spring_layout(self.G_train)
        nx.draw(self.G_train, pos)
        plt.show()


def main():
    data = Ml_100k()
    data.rating_matrix.apply_(lambda a: 1 if a >= 4 else 0)
    test_graph = Graph(data.rating_matrix)

    # print(len(list(test_graph.G_train.edges())))
    # test_graph.compute_similarity("preferential-attachment")
    # test_graph.similarities.normalize()
    # test_graph.mean_average_precision("preferential-attachment", 0.1)
    # print(test_graph.similarities)
    # test_graph.similarities.plot()
    # test_graph.plot()


if __name__ == "__main__":
    main()
