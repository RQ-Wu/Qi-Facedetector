from util import *
import sklearn.neighbors
import pickle
import os
import torch

def train(model_path=os.path.join('resources', 'knn_model', 'knn.clf'),
          n_neighbors=3, knn_algo='ball tree'):
    x = []
    y = []
    embedding_dicts = load_embedding()
    for embedding in embedding_dicts:
        x.append(embedding['embedding'])
        y.append(embedding['name'])

    knn_clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(x, y)

    if model_path is not None:
        with open(model_path, "wb") as f:
            pickle.dump(knn_clf, f)
    return knn_clf

def knn_predict(face_embedding, embeddings,
                model_path=os.path.join('resources', 'knn_model', 'knn.clf'),
                knn_threshold=0.45
                ):
    with open(model_path, "rb") as f:
        knn_clf = pickle.load(f)
    closest_distances, index = knn_clf.kneighbors(face_embedding, n_neighbors=1)
    print(int(index[0][0]))
    # print(closest_distances)

    return closest_distances[0], int(index[0][0])