import os
from os.path import join, dirname, realpath
import requests
import numpy as np
import zipfile
import io
import torch
import scipy.sparse as sp



def feature_norm(self, features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1

class Dataset(object):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset") -> None:
        self.adj_ = None
        self.features_ = None
        self.labels_ = None
        self.idx_train_ = None
        self.idx_val_ = None
        self.idx_test_ = None
        self.sens_ = None
        self.sens_idx_ = None
        self.is_normalize = is_normalize

        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.path_name = ""

    def download(self, url: str, filename: str):
        r = requests.get(url)
        assert r.status_code == 200
        open(os.path.join(self.root, self.path_name, filename), "wb").write(r.content)

    def download_zip(self, url: str):
        r = requests.get(url)
        assert r.status_code == 200
        foofile = zipfile.ZipFile(io.BytesIO(r.content))
        foofile.extractall(os.path.join(self.root, self.path_name))

    def adj(self, datatype: str = "torch.sparse"):
        # assert str(type(self.adj_)) == "<class 'torch.Tensor'>"
        if self.adj_ is None:
            return self.adj_
        if datatype == "torch.sparse":
            return self.adj_
        elif datatype == "scipy.sparse":
            return sp.coo_matrix(self.adj.to_dense())
        elif datatype == "np.array":
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError(
                "datatype should be torch.sparse, tf.sparse, np.array, or scipy.sparse"
            )

    def features(self, datatype: str = "torch.tensor"):
        if self.is_normalize and self.features_ is not None:
            self.features_ = feature_norm(self, self.features_)

        if self.features is None:
            return self.features_
        if datatype == "torch.tensor":
            return self.features_
        elif datatype == "np.array":
            return self.features_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def labels(self, datatype: str = "torch.tensor"):
        if self.labels_ is None:
            return self.labels_
        if datatype == "torch.tensor":
            return self.labels_
        elif datatype == "np.array":
            return self.labels_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_val(self, datatype: str = "torch.tensor"):
        if self.idx_val_ is None:
            return self.idx_val_
        if datatype == "torch.tensor":
            return self.idx_val_
        elif datatype == "np.array":
            return self.idx_val_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_train(self, datatype: str = "torch.tensor"):
        if self.idx_train_ is None:
            return self.idx_train_
        if datatype == "torch.tensor":
            return self.idx_train_
        elif datatype == "np.array":
            return self.idx_train_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_test(self, datatype: str = "torch.tensor"):
        if self.idx_test_ is None:
            return self.idx_test_
        if datatype == "torch.tensor":
            return self.idx_test_
        elif datatype == "np.array":
            return self.idx_test_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens(self, datatype: str = "torch.tensor"):
        if self.sens_ is None:
            return self.sens_
        if datatype == "torch.tensor":
            return self.sens_
        elif datatype == "np.array":
            return self.sens_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens_idx(self):
        if self.sens_idx_ is None:
            self.sens_idx_ = -1
        return self.sens_idx_

class Ml_100k(Dataset):
    def __init__(self, is_normalize: bool = False, root: str = "./dataset"):
        super(Ml_100k, self).__init__(is_normalize=is_normalize, root=root)
        dataset_name = "ml-100k"
        self.path_name = "ml-100k"
        if not os.path.exists(os.path.join(self.root, self.path_name)):
            os.makedirs(os.path.join(self.root, self.path_name))
        if not os.path.exists(os.path.join(self.root, self.path_name, "u.data")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/ml-100k/u.data"
            file_name = "u.data"
            self.download(url, file_name)
        if not os.path.exists(os.path.join(self.root, self.path_name, "u.user")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/ml-100k/u.user"
            file_name = "u.user"
            self.download(url, file_name)
        data = open(join(self.root, self.path_name, "u.data"))
        user_num = 943
        item_num = 1682

        rating_matrix = np.zeros([user_num, item_num])

        for line in data.readlines():
            rating = line.strip().split("\t")
            rating_matrix[int(rating[0]) - 1, int(rating[1]) - 1] = float(rating[2])

        user_info = open(join(self.root, self.path_name, "u.user"))
        user_feat = []
        for line in user_info:
            infor = line.strip().split("|")
            user_feat.append(infor[1:])

        user_sens = [0 if one[1] == "F" else 1 for one in user_feat]

        self.rating_matrix_ = rating_matrix
        self.user_sens_ = user_sens
        self.rating_matrix = torch.tensor(self.rating_matrix_)
        self.user_sens = torch.tensor(self.user_sens_)

    def rating_matrix(self, datatype: str = "torch.tensor"):
        if self.rating_matrix_ is None:
            return self.rating_matrix_
        if datatype == "torch.tensor":
            return self.rating_matrix_
        elif datatype == "np.array":
            return self.rating_matrix_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor or np.array")

    def user_sens(self, datatype: str = "torch.tensor"):
        if self.user_sens_ is None:
            return self.user_sens_
        if datatype == "torch.tensor":
            return self.user_sens_
        elif datatype == "np.array":
            return self.user_sens_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor or np.array")