import json
import pickle

import faiss
import numpy as np


def init_db(dim=1000):
    """
    restore existing vectors to database
    Args:
        dim: dimension of each vector in database

    Returns: indexing database

    """
    index = faiss.index_factory(dim, 'IDMap,Flat')

    return index


def restore_db_from_file(index_file_name):
    """
    restore database from file
    Args:
        index_file_name: index file path

    Returns:

    """
    index = faiss.read_index(index_file_name)
    return index


def insert_vector_and_id(image_id, features_vector, index_db):
    """
    insert vector in indexing database
    Args:
        image_id: id of image
        features_vector: feature vector from neural network
        index_db: indexed database

    """

    id_array = np.array([image_id], dtype=np.int64)
    vector_array = features_vector if isinstance(features_vector, np.ndarray) else np.array(features_vector)
    index_db.add_with_ids(vector_array, id_array)


def find_vector(features_vector, index_db, n_results=1):
    """
    find vector in database
    Args:
        features_vector: feature vector from neural network
        index_db: indexed database
        n_results: find n nearest vector

    Returns: n nearest vectors

    """
    _, index_res = index_db.search(features_vector, n_results)
    result_data = index_res[0].tolist()

    return result_data


def indexing_db(index):
    """
    indexing database
    Args:
        index: database

    """
    index.train()


def dump_db(index_db, save_index_file_path):
    """Dump data in file
        Args:
            index - database module
            index_file - file to save feature vector indices
    """
    faiss.write_index(index_db, save_index_file_path)
