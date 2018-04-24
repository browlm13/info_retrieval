import json
import logging
import sys
import os
import math

# my lib
from src import file_io

# external
import glob
import pandas as pd
import numpy as np


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))


def load_document_frequency_dicts(indexed_directory_name):
    document_frequency_dict_file_template = file_io.get_template('document_frequency_dict_file_path') % \
                                            (indexed_directory_name, '*')

    document_id_term_frequency_dict = {}
    for dfd_path in glob.glob(document_frequency_dict_file_template):
        with open(dfd_path) as json_data:
            dfd = json.load(json_data)
            doc_id = str(dfd['document_id'])
            doc_term_freq_dict = dfd['term_frequency_dict']
            document_id_term_frequency_dict[doc_id] = doc_term_freq_dict

    return document_id_term_frequency_dict


def get_document_term_frequency_matrix(indexed_directory_name, write=True):

    # check if document term frequency matrix has already been created
    matrix_file = file_io.get_path("document_term_frequency_matrix_file_path", None, force=True)

    # if the document term frequency matrix has already been created already exists load and return
    if os.path.isfile(matrix_file):
        logger.info("Accessing document term frequency matrix already in existence at: %s" % matrix_file)
        return pd.read_csv(matrix_file, index_col="docID") #, dtype={'docID': np.int64, 'document text': str})

    # if the document term frequency matrix does not exist, construct it from document frequency json files
    id_tf_dict = load_document_frequency_dicts(indexed_directory_name)

    unique_words = set()
    for doc_id, tf_dict in id_tf_dict.items():
        unique_words = unique_words | set(tf_dict.keys())

    doc_freq_matrix = pd.DataFrame(columns=unique_words) # , index=id_tf_dict.keys())
    doc_freq_matrix.index.name = "docID"
    for doc_id, tf_dict in id_tf_dict.items():
        terms, freqs = zip(*tf_dict.items())
        df = pd.DataFrame(data=[freqs], columns=terms, index=[int(doc_id)])
        #df.index.name = "docID"
        doc_freq_matrix = pd.concat([doc_freq_matrix, df], join='outer')
    doc_freq_matrix = doc_freq_matrix.fillna(value=0)

    # sort by docID
    doc_freq_matrix = doc_freq_matrix.sort_index() # by='docID')

    # set index column name to docID
    #doc_freq_matrix.index.name = 'docID'

    # write to csv
    if write:
        logger.info("Writing Document Term Frequency Matrix")
        matrix_file = file_io.get_path("document_term_frequency_matrix_file_path", None, force=True)
        doc_freq_matrix.to_csv(matrix_file)

    return doc_freq_matrix

def document_vector_matrix_and_index_dicts(doc_freq_matrix_dataFrame):
    """
    :param doc_freq_matrix_dataFrame: pandas dataFrame representation of the document frequency matrix which includes full words
        as column names and a column for the docID
    :return: document_vector_matrix, word2col, docID2row - 2d numpy array matrix with rows as document term frequency vectors, and python
        dictionaries to convert words to column indices and docIDs to row indices
    """
    document_vector_matrix = doc_freq_matrix_dataFrame.as_matrix()
    word2col = zip(doc_freq_matrix_dataFrame.columns.values, range(doc_freq_matrix_dataFrame.shape[1]))
    docID2row = zip(doc_freq_matrix_dataFrame.index.values, range(doc_freq_matrix_dataFrame.shape[0]))

    return document_vector_matrix, dict(docID2row), dict(word2col)


def cosine_similarity(document_vector_1, document_vector_2):
    """
    :param document_vector_1:
    :param document_vector_2:
    :return: float - normalized dot product between vectors (angle)
    """
    dot_product = np.dot(document_vector_1, document_vector_2)
    magnitude_1 = np.linalg.norm(document_vector_1)
    magnitude_2 = np.linalg.norm(document_vector_1)

    return dot_product/(magnitude_1*magnitude_2)


def ranked_cosine_similarity(query_vector, document_vector_matrix):
    """

    :param query_vector:
    :param document_vector_matrix:
    :return: ordered numpy array with indices of row vectors with the highest scoring cosine similarity to query
    """

    # compute cosine similarity of all document vectors in matrix with query vector
    func = lambda row : cosine_similarity(query_vector, row)
    cossim_vector = np.apply_along_axis(func, 1, document_vector_matrix)

    # find ranked indices in matrix by cosine similarity
    inplace_max_indices = np.argsort(cossim_vector)
    reversed_index_range_vector = np.array(range(document_vector_matrix.shape[0]))[::-1]
    indices_to_sorted_max_indices = np.argsort(inplace_max_indices)
    max_indices = np.argsort(reversed_index_range_vector[indices_to_sorted_max_indices])

    return max_indices


def cluster_pruning_leader_follower_dict(doc_freq_matrix_dataFrame):
    """
    Select docIds for leaders and followers and format in python dictionary
    :param doc_freq_matrix_dataFrame:
    :return: return sqrt(N) leaders with sqrt(N) followers as python dictionary with keys as leader docIDs and values
            as list of follower docIDs
    """
    document_vector_matrix, docID2row, word2col = document_vector_matrix_and_index_dicts(doc_freq_matrix_dataFrame)

    # choose n random document vectors indices
    N = doc_freq_matrix_dataFrame.shape[0]
    sqrtN = int(math.sqrt(N))
    leader_indices = np.random.randint(low=0, high=N, size=sqrtN, dtype=int)

    # find each leaders top sqrtN followers using cosine similarity
    find_follower_list = lambda leader_idx: ranked_cosine_similarity(document_vector_matrix[leader_idx],document_vector_matrix)[:sqrtN]
    follower_indices_matrix = np.apply_along_axis(find_follower_list,0,leader_indices)

    # create leader follower dictionary
    leader_dictionary = {}
    row2docID = {v:k for k,v in docID2row.items()}
    for i,lidx in enumerate(leader_indices.tolist()):
        leader_docID = row2docID[lidx]
        follower_docIDs = [row2docID[idx] for idx in follower_indices_matrix[i].tolist()]
        leader_dictionary[leader_docID] = follower_docIDs

    # return leader follower dictonary
    return leader_dictionary

#def
