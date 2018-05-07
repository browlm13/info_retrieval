import json
import logging
import sys
import os
import math

# my lib
from src import file_io
from src import text_processing

# external
import glob
import pandas as pd
import numpy as np


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
# logger.addHandler(logging.StreamHandler(sys.stdout))


def load_document_frequency_dicts(indexed_directory_name):
    document_frequency_dict_file_template = file_io.get_template('document_term_frequency_dictionary_file_path') % \
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


# join hash_url_list_map and hash_id_map (stupid name doc prefix) on hash for url_id connection
def get_docID2url_map():

    hash_id_map_file = file_io.get_path("hash_id_map_file", None, force=True)
    with open(hash_id_map_file) as json_data:
        hash_id_map = json.load(json_data)


    #hash_id_df = pd.DataFrame(hash_id_tuple_list, columns=['hash', 'id'])

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

def cluster_pruning_matrix_and_maps(doc_freq_matrix_dataFrame):
    """
    Select docIds for leaders and followers and format in python dictionary
    :param doc_freq_matrix_dataFrame:
    :return: return sqrt(N) leaders with sqrt(N) followers as python dictionary with keys as leader docIDs and values
            as list of follower docIDs
    """
    logger.info("Cluster Pruning - Preprocessing")
    document_vector_matrix, docID2row, word2col = document_vector_matrix_and_index_dicts(doc_freq_matrix_dataFrame)

    # choose n random document vectors indices
    N = doc_freq_matrix_dataFrame.shape[0]
    sqrtN = int(math.sqrt(N))
    cluster_size = sqrtN + 1 # sqrtN + 1 for leader
    random_indices = np.random.randint(low=0, high=N, size=sqrtN, dtype=int) # sqrtN random indices for sqrtN leaders
    # (note: not necessarily leader indices)

    # find each leaders top sqrtN followers using cosine similarity
    # cluster matrix elements represent row indices in numpy document term frequency matrix
    # follower list will start with leader as first index if leader is in matrix / is only equal document
    find_cluster_array = lambda random_idx: ranked_cosine_similarity(document_vector_matrix[random_idx],
                                                                     document_vector_matrix)[:cluster_size]

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
    # leader2cluster_id_dict = dict(zip(leader_ids, cluster_id_lists))  # leader maps to entire cluster id

    # create leader matrix to save - used for quicker comparisons with query
    # get leader vectors as matrix
    leader_indices = cluster_matrix[:,0]
    leader_document_vector_matrix = document_vector_matrix[leader_indices]  # used for best quick comparison with query

    # go from leader_document_vector_matrix index with highest cosine similarity
    # to leader and follower id list for quick access

    # get leader  document vector matrix row to docID
    leader_row_2_cluster_ids = dict(zip(range(0,leader_indices.shape[0]), cluster_id_lists))

    # get docID url map
    # docID2url = get_docID2url_map()

    # get docID title map

    #
    # TODO: title matrix
    #

    # return leader document vector matrix and maps
    return leader_document_vector_matrix, leader_row_2_cluster_ids #, docID2url

    """
    # turn cluster_id_lists into url lists
    docIDlist_2_urllist = lambda docID_list: [docID2url[docID] for docID in docID_list]
    cluster_urls_lists = [docIDlist_2_urllist(docID_list) for docID_list in id_cluster_matrix.tolist()]

    # go from leader row directly to list of urls
    leader_row_2_cluster_urls = dict(zip(range(0, leader_indices.shape[0]), cluster_urls_lists))
    """


    """
    # return leader follower dictonary
    if to_json:
        leader_dict_json = {str(k) : v for k,v in leader_dict.items()}
        return leader_dict_json
    return leader_dict
    """


def cluster_pruning_leader_follower_dict(doc_freq_matrix_dataFrame, to_json=False):
    """
    Select docIds for leaders and followers and format in python dictionary
    :param doc_freq_matrix_dataFrame:
    :return: return sqrt(N) leaders with sqrt(N) followers as python dictionary with keys as leader docIDs and values
            as list of follower docIDs
    """
    logger.info("Cluster Pruning - Preprocessing")
    document_vector_matrix, docID2row, word2col = document_vector_matrix_and_index_dicts(doc_freq_matrix_dataFrame)

    # choose n random document vectors indices
    N = doc_freq_matrix_dataFrame.shape[0]
    sqrtN = int(math.sqrt(N))
    cluster_size = sqrtN + 1 # sqrtN + 1 for leader
    random_indices = np.random.randint(low=0, high=N, size=sqrtN, dtype=int) # sqrtN random indices for sqrtN leaders
    # (note: not necessarily leader indices)

    # find each leaders top sqrtN followers using cosine similarity
    # follower list will start with leader as first index if leader is in matrix / is only equal document
    find_cluster_array = lambda random_idx: ranked_cosine_similarity(document_vector_matrix[random_idx],
                                                                     document_vector_matrix)[:cluster_size]

    vfunc = np.vectorize(find_cluster_array, signature='()->(sqrtN)')
    cluster_matrix = vfunc(random_indices)

    # turn cluster matrix into id cluster matrix (use docIDs instead of element indices)
    row2docID = {v: k for k, v in docID2row.items()}
    index2docID = lambda index: row2docID[index]
    vfunc = np.vectorize(index2docID)
    id_cluster_matrix = vfunc(cluster_matrix)

    # create leader follower dictonary (with docIDs)
    leaders = id_cluster_matrix[:,0] # use first column as leaders
    follower_matrix = id_cluster_matrix[:, 1:] # everything else
    follower_lists = follower_matrix.tolist()
    leader_dict = dict(zip(leaders, follower_lists))

    # return leader follower dictonary
    if to_json:
        leader_dict_json = {str(k) : v for k,v in leader_dict.items()}
        return leader_dict_json
    return leader_dict


def save_leader_follower_dictionary(doc_freq_matrix_dataFrame):
    logger.info("Saving Leader Follower File...")
    leader_follower_docID_json = cluster_pruning_leader_follower_dict(doc_freq_matrix_dataFrame, to_json=True)
    # write file
    file_io.save('leader_follower_file_path', leader_follower_docID_json, None)


def load_leader_follower_dictionary():

    logger.info("Loading Leader Follower File...")
    leader_follower_docID_dict_file_path = file_io.get_path('leader_follower_file_path', None)
    if leader_follower_docID_dict_file_path is not None:
        with open(leader_follower_docID_dict_file_path) as json_data:
            leader_follower_dict_json = json.load(json_data)
            # cast docID key to int
            leader_follower_dict = {int(k) : v for k,v in leader_follower_dict_json.items()}
        return leader_follower_dict
    else:
        logger.error("Leader Follower File Not Found")


def query_to_vector(raw_query, word2col):
    # create empty query vector
    query_vector = np.zeros(len(word2col))

    # tokenize query
    query_tokens = text_processing.plain_text_to_tokens(raw_query)  # , stopwords file)

    # update term frequencies of query vector
    for token in query_tokens:
        column_index = word2col[token]
        query_vector[column_index] += 1

    return query_vector


def query_to_vector_slow(raw_query):

    # all that is needed is word2col dictonary
    word2col_file_path = file_io.get_path('word2col_file_path', None)
    with open(word2col_file_path) as json_data:
        word2col = json.load(json_data)

    # create empty query vector
    query_vector = np.zeros(len(word2col))

    # tokenize query
    query_tokens = text_processing.plain_text_to_tokens(raw_query) # , stopwords file)

    # update term frequencies of query vector
    for token in query_tokens:
        column_index = word2col[token]
        query_vector[column_index] += 1

    return query_vector


def vector_to_tokens(query_vector, col2word):
    token_list = []
    word_indices = np.nonzero(query_vector)[0] # column indices
    for i in word_indices:
        token_list.append(col2word[i])
    return token_list


def vector_to_tokens_slow(query_vector):
    # all that is needed is col2word dictonary
    document_vector_matrix, docID2row, word2col = document_vector_matrix_and_index_dicts(doc_freq_matrix_dataFrame)

    col2word = {v: k for k, v in word2col.items()}
    token_list = []
    word_indices = np.nonzero(query_vector)[0] # column indecies
    for i in word_indices:
        token_list.append(col2word[i])

    return token_list

def vector_to_tokens_slow(query_vector, doc_freq_matrix_dataFrame):
    document_vector_matrix, docID2row, word2col = document_vector_matrix_and_index_dicts(doc_freq_matrix_dataFrame)

    col2word = {v: k for k, v in word2col.items()}
    token_list = []
    word_indices = np.nonzero(query_vector)[0] # column indecies
    for i in word_indices:
        token_list.append(col2word[i])

    return token_list

# def postings list <-> document term frequency matrix