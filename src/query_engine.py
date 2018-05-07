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
#   Query Engine
#
"""
load: 
    -leader_document_vector_matrix
    -title_document_vector_matrix
    -matrix_maps
    -convert query to vector using word2col
    -compare with leaders / use leader row2 cluster id map
    -0.25 query score thing with title matrix
"""

class QueryEngine:

    def __init__(self, output_directory_name):

        # TODO: should build matrices and maps if output_directory_name does not exist

        # TODO: should not need following parameter
        self.output_directory_name = output_directory_name

        # load
        # self.load_leader_document_vector_matrix()
        # self.load_title_document_vector_matrix()
        # self.load_matrix_maps()

        # load leader document vector matrix
        ldvm_file_path = file_io.get_path("leader_document_vector_matrix_file_path", [output_directory_name])
        self.leader_document_vector_matrix = np.load(ldvm_file_path)

        # load title document vector matrix
        tdvm_file_path = file_io.get_path("title_document_vector_matrix_file_path", [output_directory_name])
        self.title_document_vector_matrix = np.load(tdvm_file_path)

        # load matrix maps
        matrix_maps_file_path = file_io.get_path("matrix_maps_file_path", [output_directory_name])
        with open(matrix_maps_file_path, 'rb') as pickle_file:
            self.matrix_maps = pickle.load(pickle_file)

        self.word2col = self.matrix_maps['word2col']
        self.col2word = self.matrix_maps['col2word']    # tmp not needed
        self.leader_row_2_cluster_indices = self.matrix_maps['leader_row_2_cluster_indices']
        self.leader_row_2_cluster_ids = self.matrix_maps['leader_row_2_cluster_ids']
        self.docID2url = self.matrix_maps['docID2url']

    def query_to_vector(self, raw_query):
        # create empty query vector
        query_vector = np.zeros(len(self.word2col))

        # tokenize query
        query_tokens = text_processing.plain_text_to_tokens(raw_query)  # , stopwords file)

        # update term frequencies of query vector
        for token in query_tokens:
            try:
                column_index = self.word2col[token]
                query_vector[column_index] += 1
            except KeyError:
                logger.info("Query word not found in index: %s (stemmed)" % token)

        return query_vector

    def vector_to_tokens(self, query_vector):
        token_list = []
        word_indices = np.nonzero(query_vector)[0]  # column indices
        for i in word_indices:
            token_list.append(self.col2word[i])
        return token_list

    def search(self, raw_query):

        # convert query to vector
        query_vector = self.query_to_vector(raw_query)
        # tokens = self.vector_to_tokens(query_vector)
        # print(tokens)

        # find nearest leader document vector to query vector
        nearest_leader_row = document_vector_operations.ranked_cosine_similarity(query_vector,
                                                                                 self.leader_document_vector_matrix)[0]

        # find the selected clusters document id list and urls
        cluster_indices = np.array(self.leader_row_2_cluster_indices[nearest_leader_row])
        cluster_ids = np.array(self.leader_row_2_cluster_ids[nearest_leader_row])

        # tmp slow
        # cluster_ids = self.leader_row_2_cluster_ids[nearest_leader_row]
        # cluster_urls = [self.docID2url[docID] for docID in cluster_ids]
        # print(cluster_urls)

        # add .25 to scores of documents in cluster where the titles contain words in the query
        # the dot product of the query vector and the title vector are greater than 1

        # find title vectors of cluster documents
        title_vectors = self.title_document_vector_matrix[cluster_indices]

        title_tokens = [self.vector_to_tokens(tv) for tv in title_vectors]
        print(title_tokens)

        # take the dot product of the query vector against each title vector
        dot_results = np.dot(title_vectors, query_vector)

        # multiply by 0.25 - skip same difference

        # sort by max indices and apply this to the cluster ids for ranked results (negative results for reverse order)
        ranked_result_ids = cluster_ids[np.argsort(-dot_results)]
        # ranked_result_urls = [self.docID2url[docID] for docID in ranked_result_ids]


        #display
        display_strings = self.ranked_results_display_strings(ranked_result_ids)
        for ds in display_strings:
            print(ds)


        # get non zero indices
        # non_zero_indices = np.nonzero(dot_results)

    def get_title(self, docID):         # TODO: Merge title databases so outputdir is not needed
        title_path = file_io.get_path('document_title_file_path', [self.output_directory_name, docID])
        with open(title_path) as json_data:
            dtd = json.load(json_data)
            doc_title = dtd['title']
        return doc_title


    def ranked_results_display_strings(self, ranked_result_ids):
        ranked_result_urls = [self.docID2url[docID] for docID in ranked_result_ids]
        ranked_result_titles = [self.get_title(docID) for docID in ranked_result_ids]

        urls_and_titles = zip(ranked_result_urls, ranked_result_titles)
        title_or_none = lambda title: "NO TITLE" if title == None else title
        format_urls_and_titles = lambda url, title: url + '\n' + title_or_none(title) + '\n'
        display_strings = [format_urls_and_titles(unt[0], unt[1]) for unt in urls_and_titles]
        return display_strings