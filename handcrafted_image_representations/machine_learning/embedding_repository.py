import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
from umap import UMAP
from sklearn.cluster import MiniBatchKMeans

from handcrafted_image_representations.machine_learning.image_embedding import ImageEmbedding
from handcrafted_image_representations.utils.utils import load_dict, save_dict


class EmbeddingRepository:
    def __init__(self, 
                 path_to_store=None,
                 update_cycle=100,
                 aggregator="bag_of_words", 
                 complexity=1024, 
                 feature="hsv-sift",
                 sampling_method="dense",
                 sampling_window=16,
                 sampling_step=16,
                 image_size_width=128,
                 image_size_height=128,
                 ):
        
        print("[INFO] Initializing Repository for {}".format(path_to_store))
        self.path = path_to_store
        if path_to_store is not None:
            os.makedirs(self.path, exist_ok=True)
            self.db_path = os.path.join(self.path, "db.json")
            self.data_path = os.path.join(self.path, "data.npy")
            self.model_path = os.path.join(self.path, "model")

        self.update_cycle = update_cycle

        self.db = None
        self.data = None
        self.length = 0

        self.model = ImageEmbedding(
            aggregator=aggregator, 
            complexity=complexity, 
            feature=feature,
            sampling_method=sampling_method,
            sampling_window=sampling_window,
            sampling_step=sampling_step,
            image_size_width=image_size_width,
            image_size_height=image_size_height,
        )

        self.load()

    def load_data(self, tag):
        return tag.load_data()
    
    def import_fitted_model(self, model_path):
        self.model.load(model_path)

    def load(self):
        if self.path is None:
            return
        if os.path.isdir(self.model_path) and self.model is not None:
            self.model.load(self.model_path)

        if not os.path.isfile(self.db_path):
            return
        if not os.path.isfile(self.data_path):
            return
        
        self.db = load_dict(self.db_path)
        self.data = np.load(self.data_path)
        assert len(self.db) == self.data.shape[0], "DataBase and Data does not match"
        print("[INFO] Loaded {} representations".format(len(self.db)))
        self.length = len(self.db)

    def save(self):
        if self.path is None:
            return
        save_dict(self.db, self.db_path)
        np.save(self.data_path, self.data)
        self.model.save(self.model_path)
        self.length = len(self.db)

    def store(self, list_of_identifier, representations):
        if type(list_of_identifier) is not list:
            list_of_identifier = [list_of_identifier]
        if len(representations.shape) == 1:
                representations = np.reshape(representations, (1, -1))
        assert len(list_of_identifier) == representations.shape[0], "Length missmatch"
        if self.db is None or self.data is None:
            self.db = {ident: i for i, ident in enumerate(list_of_identifier)}
            self.data = representations
        else:
            for ident in list_of_identifier:
                if ident in list_of_identifier:
                    self.db[ident] = len(self.db)
                else:
                    self.db[ident] = len(self.db)
            self.data = np.concatenate([self.data, representations])
        
        if len(self.db) - self.length > self.update_cycle:
            self.save()

    def fit(self, tags):
        list_of_identifier = [tag.id for tag in tags]
        x = self.model.fit_transform(tags)
        x = np.concatenate(x, axis=0)
        self.store(list_of_identifier, x)
    
    def identifier_exists(self, identifier):
        if self.db is None:
            return False
        if identifier not in self.db:
            return False
        return True

    def register_sample(self, identifier, image):
        if self.identifier_exists(identifier):
            return
        x = self.model.transform(image)[0, :]
        self.store(identifier, x)
        logging.info("Ident: {} registered".format(identifier))
        return x
    
    def _extract_tag(self, tag):
        return self.model.transform(self.load_data(tag))[0, :]
    
    def register_batch(self, tags):
        tags_to_process = [tag for tag in tags if not self.identifier_exists(tag.id)]
        if len(tags_to_process) == 0:
            print("[INFO] All already processed")
            return
        t0 = time()
        idents = [tag.id for tag in tags_to_process]
        print("[INFO] Computing Reps for {} Points...".format(len(tags_to_process)))
        with Pool() as p:
            reps = p.map(self._extract_tag, tags_to_process)
        self.store(idents, np.array(reps))
        delta_t = np.round(time() - t0, 4)
        print("[INFO] done in {}s ({}tags/s)".format(delta_t, np.round(len(tags_to_process) / delta_t, 4)))

    def load_sample(self, identifier):
        return self.data[self.db[identifier], :]
    
    def transform(self, identifier, image):
        if identifier is None:
            return self.model.transform(image)[0, :]
        if not self.identifier_exists(identifier):
            return self.register_sample(identifier, image)
        return self.load_sample(identifier)
    
    def transform_tag(self, tag):
        if not self.identifier_exists(tag.id):
            return self.register_sample(tag.id, self.load_data(tag))
        return self.load_sample(tag.id)
    
    def project_tag(self, tag):
        x_tag = self.transform_tag(tag)
        projection = UMAP(n_components=2)
        x_proj = projection.fit_transform(self.data)
        x_tag = projection.transform(x_tag.reshape(1, -1))

        df = pd.DataFrame({"cls": "bg", "x1": x_proj[:, 0], "x2": x_proj[:, 1]})
        df_tag = pd.DataFrame({"cls": "Tag", "x1": x_tag[0, 0], "x2": x_tag[0, 1]}, index=[0])
        df = pd.concat([df, df_tag])

        plt.title("Distribution")
        sns.scatterplot(data=df, x="x1", y="x2", hue="cls")
        plt.show()
    
    def show(self, tag_id_to_class=None, path_to_store=None):
        assert self.data is not None, "No Representations are computed."

        projection = UMAP(n_components=2)
        x_proj = projection.fit_transform(self.data)

        list_of_classes = ["None"] * len(self.db)
        if tag_id_to_class is not None:
            for tag_id, index in self.db.items():
                if tag_id not in tag_id_to_class:
                    continue
                list_of_classes[index] = tag_id_to_class[tag_id]

        df = pd.DataFrame({
            "cls": list_of_classes,
            "x1": x_proj[:, 0],
            "x2": x_proj[:, 1]
        })

        plt.title("Distribution")
        sns.scatterplot(data=df, x="x1", y="x2", hue="cls")
        if path_to_store is None:
            plt.show()
        else:
            plt.savefig(path_to_store)
            plt.close()

    def query(self, image, n=3, sim="euclidean"):
        assert self.data is not None, "No Representations are computed."
        inv_db = {v: k for k, v in self.db.items()}
        x_trans = self.transform(None, image)
        if sim == "euclidean":
            distances = np.sqrt(np.sum(np.square(self.data - x_trans), axis=1))
        elif sim == "manhattan":
            distances = np.sqrt(np.sum(np.abs(self.data - x_trans), axis=1))
        elif sim == "cosine":
            norm_x = np.linalg.norm(self.data, axis=1) * np.linalg.norm(x_trans)
            norm_x[norm_x == 0] = 1e-6
            distances = np.dot(self.data, x_trans.T)/(norm_x)
        else:
            raise ValueError("Unknown Similarity Metric")
        sorted_indices = np.argsort(distances)
        return [inv_db[i] for i in sorted_indices[:n]]
    
    def sample(self, n):
        assert self.data is not None, "No Representations are computed."
        inv_db = {v: k for k, v in self.db.items()}

        clustering = MiniBatchKMeans(n_clusters=n, n_init="auto")
        clustering.fit(self.data)
        cluster_assignments = clustering.labels_

        cluster_indices = {}
        for idx, label in enumerate(cluster_assignments):
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(idx)

        # Randomly sample one instance from each cluster
        sampled_indices = []
        for _, indices in cluster_indices.items():
            sampled_indices.append(np.random.choice(indices))

        return [inv_db[i] for i in sampled_indices]

