import json
import logging
import sys
import os
import math

# my lib
from src import file_io
from src import text_processing
from src import document_vector_operations

# external
import glob
import pandas as pd
import numpy as np
import pickle

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# logger.addHandler(logging.FileHandler("output/output_log.txt"))
# logger.addHandler(logging.StreamHandler(sys.stdout))

#
#   Build Matrices and Maps
#

# builds numpy matrices and maps from scraped directory
# matrices : full document vector matrix, title document vector matrix, leader document vector matrix
# maps : row2docID++inv, word2col+inv,  leader_row_2_cluster_ids,
# notes: make title document vector matrix the same dimensions as the other document vectors in same order for map
#             reuse
# save compactly

def build_matrices_and_maps(indexed_directory_name_list):

    output_directory_name = '_'.join(indexed_directory_name_list)

    # create term_frequency_dictionaries and find unique_words
    combined_full_id_tf_dict = {}
    combined_title_id_tf_dict = {}
    for indexed_directory_name in indexed_directory_name_list:
        full_id_tf_dict = load_document_term_frequency_dictionaries(indexed_directory_name)
        title_id_tf_dict = load_title_document_id_term_frequency_dictionaries(indexed_directory_name)
        combined_full_id_tf_dict.update(full_id_tf_dict)
        combined_title_id_tf_dict.update(title_id_tf_dict)
    unique_words = find_all_unique_words([combined_full_id_tf_dict, combined_title_id_tf_dict])

    # create full, title and leader dvms and maps
    full_document_vector_matrix, docID2row, word2col = matrix_and_maps(combined_full_id_tf_dict, unique_words)
    title_document_vector_matrix, _, _ = matrix_and_maps(combined_title_id_tf_dict, unique_words)
    leader_document_vector_matrix, leader_row_2_cluster_ids = \
        cluster_pruning_matrix_and_maps(full_document_vector_matrix, docID2row)

    # save matrices and maps
    file_io.save('full_document_vector_matrix_file_path', full_document_vector_matrix,
                 [output_directory_name], output_type='numpy_array')
    file_io.save('title_document_vector_matrix_file_path', title_document_vector_matrix,
                 [output_directory_name], output_type='numpy_array')
    file_io.save('leader_document_vector_matrix_file_path', leader_document_vector_matrix,
                 [output_directory_name], output_type='numpy_array')

    # save all maps in one file
    matrix_maps = {
        'docID2url': get_docID2url_map(),
        'row2docID': {v: k for k, v in docID2row.items()},
        'docID2row': docID2row,
        'col2word': {v: k for k, v in word2col.items()},
        'word2col': word2col
    }
    file_io.save('matrix_maps_file_path', matrix_maps, [output_directory_name], output_type='pickle_dict')


def load_document_term_frequency_dictionaries(indexed_directory_name):
    logger.info("Loading Document Frequency Dictionaries")
    document_frequency_dictionary_file_template = \
        file_io.get_template('document_term_frequency_dictionary_file_path') % (indexed_directory_name, '*')
    document_id_term_frequency_dictionary = {}
    for dfd_path in glob.glob(document_frequency_dictionary_file_template):
        with open(dfd_path) as json_data:
            dfd = json.load(json_data)
            doc_id = str(dfd['document_id'])
            doc_term_freq_dict = dfd['term_frequency_dict']
            document_id_term_frequency_dictionary[doc_id] = doc_term_freq_dict
    return document_id_term_frequency_dictionary


def load_title_document_id_term_frequency_dictionaries(indexed_directory_name):
    logger.info("Loading Title Frequency Dictionaries")
    document_title_dictionary_file_template = \
        file_io.get_template('document_title_file_path') % (indexed_directory_name, '*')
    document_id_term_frequency_dictionary = {}
    for dtd_path in glob.glob(document_title_dictionary_file_template):
        with open(dtd_path) as json_data:
            dtd = json.load(json_data)
            doc_id = str(dtd['document_id'])
            doc_title = dtd['title']

            # No title
            if doc_title is None:
                doc_title = 'NO_TITLE'

            title_tokens = text_processing.plain_text_to_tokens(doc_title)  # ,stopwords_file)
            doc_term_freq_dict = text_processing.word_frequency_dict(title_tokens)
            document_id_term_frequency_dictionary[doc_id] = doc_term_freq_dict
    return document_id_term_frequency_dictionary


def cluster_pruning_matrix_and_maps(full_document_vector_matrix, docID2row):
    """
    Select docIds for leaders and followers and format in python dictionary
    :return: return sqrt(N) leaders with sqrt(N) followers as python dictionary with keys as leader docIDs and values
            as list of follower docIDs
    """
    logger.info("Cluster Pruning - Preprocessing")
    #document_vector_matrix, docID2row, word2col = document_vector_matrix_and_index_dicts(doc_freq_matrix_dataFrame)


    # choose n random document vectors indices
    N = len(docID2row)
    sqrtN = int(math.sqrt(N))
    cluster_size = sqrtN + 1 # sqrtN + 1 for leader
    random_indices = np.random.randint(low=0, high=N, size=sqrtN, dtype=int) # sqrtN random indices for sqrtN leaders
    # (note: not necessarily leader indices)

    # find each leaders top sqrtN followers using cosine similarity
    # cluster matrix elements represent row indices in numpy document term frequency matrix
    # follower list will start with leader as first index if leader is in matrix / is only equal document
    find_cluster_array = lambda random_idx: \
        document_vector_operations.ranked_cosine_similarity(full_document_vector_matrix[random_idx],
                                                                     full_document_vector_matrix)[:cluster_size]

    vfunc = np.vectorize(find_cluster_array, signature='()->(sqrtN)')
    cluster_matrix = vfunc(random_indices)

    # turn cluster matrix into id cluster matrix (use docIDs as elements instead of element indices as element indices)
    row2docID = {v: k for k, v in docID2row.items()}
    index2docID = lambda index: row2docID[index]
    vfunc = np.vectorize(index2docID)
    id_cluster_matrix = vfunc(cluster_matrix)

    # create leader follower dictionary (with docIDs)
    leader_ids = id_cluster_matrix[:,0] # use first column as leaders
    # follower_id_matrix = id_cluster_matrix[:, 0:] # follower id matrix includes leader for leader follower dictionary
    cluster_id_lists = id_cluster_matrix.tolist() # includes leader id too

    # create leader matrix to save - used for quicker comparisons with query
    # get leader vectors as matrix
    leader_indices = cluster_matrix[:,0]
    leader_document_vector_matrix = full_document_vector_matrix[leader_indices]  # used for best quick comparison with query

    # go from leader_document_vector_matrix index with highest cosine similarity
    # to leader and follower id list for quick access

    # get leader  document vector matrix row to docID
    leader_row_2_cluster_ids = dict(zip(range(0,leader_indices.shape[0]), cluster_id_lists))

    # return leader document vector matrix and maps
    return leader_document_vector_matrix, leader_row_2_cluster_ids


def display_loading_bar(completion_percentage, process_title):
    bar_length = 100  # in characters
    divisor = int(100 / bar_length)
    loading_bar_string = '[' + '*' * int(completion_percentage / divisor) + \
                         '_' * (bar_length - int(completion_percentage / divisor)) + ']'
    completion_string = ('Loading %s...\n %s%% Complete.' % (
        process_title, str(int(completion_percentage)))) + loading_bar_string
    logger.info(completion_string)


def find_all_unique_words(id_tf_dict_list):
    logger.info("Finding all unique words...")
    unique_words = set()
    for id_tf_dict in id_tf_dict_list:
        for doc_id, tf_dict in id_tf_dict.items():
            unique_words = unique_words | set(tf_dict.keys())
    return unique_words




def matrix_and_maps(document_id_term_frequency_dictionary, unique_words):

    logger.info("Filling Pandas DataFrame...")
    doc_freq_matrix = pd.DataFrame(columns=unique_words)  # , index=id_tf_dict.keys())
    doc_freq_matrix.index.name = "docID"

    num_tf_dicts_added = 0
    total_num_tf_dicts = len(document_id_term_frequency_dictionary)
    for doc_id, tf_dict in document_id_term_frequency_dictionary.items():
        terms, freqs = zip(*tf_dict.items())
        df = pd.DataFrame(data=[freqs], columns=terms, index=[int(doc_id)])
        doc_freq_matrix = pd.concat([doc_freq_matrix, df], join='outer')

        # log completion percentage and loading bar
        process_title = "Document TF index files into Pandas DataFrame"
        num_tf_dicts_added += 1
        completion_percentage = 100 * float(num_tf_dicts_added / total_num_tf_dicts)
        display_loading_bar(completion_percentage, process_title)

    logger.info("Changing NA values to 0's...")
    doc_freq_matrix = doc_freq_matrix.fillna(value=0)

    # sort by docID
    logger.info("Sorting Document Term Frequency Matrix rows by docID ascending...")
    df_dtfm = doc_freq_matrix.sort_index()

    logger.info("Creating Numpy Representation...")
    np_dtfm = df_dtfm.as_matrix()

    logger.info("Creating Maps...")
    logger.info("Creating word2col Map...")
    word2col = dict(zip(df_dtfm.columns.values, range(df_dtfm.shape[1])))
    logger.info("Creating docID2row Map...")
    docID2row = dict(zip(df_dtfm.index.values, range(df_dtfm.shape[0])))

    return np_dtfm, docID2row, word2col

def get_docID2url_map():

    hash_id_map_file = file_io.get_path("hash_id_map_file", None, force=True)
    with open(hash_id_map_file) as json_data:
        hash_id_map = json.load(json_data)

    hash_url_list_map_file = file_io.get_path("hash_url_list_map_file", None, force=True)
    with open(hash_url_list_map_file) as json_data:
        hash_url_list_map = json.load(json_data)

    # take all urls
    # docID_url_map = {hash_id_map[hash]:hash_url_list_map[hash] for hash in hash_id_map.keys()}

    # take only first url
    docID_url_map = {hash_id_map[hash]: hash_url_list_map[hash][0] for hash in hash_id_map.keys()}
    return docID_url_map

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