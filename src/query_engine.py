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

    def __init__(self, output_directory_name, search_type="full_search", weighting_type="tf"):

        # TODO: should build matrices and maps if output_directory_name does not exist

        self.output_directory_name = output_directory_name
        self.search_type = search_type
        self.weighting_type = weighting_type

        # TODO: load based on search type, don't load for each individual search
        self.load_maps()

        # load matrices based on search type and weighting_type
        if search_type == "cluster_pruning":
            # TODO: impliment tfidf weighting option
            self.load_matrices(['leader_document_vector_matrix', 'title_document_vector_matrix'])
        if search_type == "full_search":
            self.load_matrices(['title_document_vector_matrix'])
            if weighting_type == "tf":
                self.load_matrices(['full_document_vector_matrix'])
            if weighting_type == "tfidf":
                self.load_matrices(['tfidf_matrix'])
                self.full_document_vector_matrix = self.tfidf_matrix

    def load_maps(self):

        # load matrix maps
        matrix_maps_file_path = file_io.get_path("matrix_maps_file_path", [self.output_directory_name])
        with open(matrix_maps_file_path, 'rb') as pickle_file:
            self.matrix_maps = pickle.load(pickle_file)

        # loading without optimization loads all
        self.word2col = self.matrix_maps['word2col']      # query_to_vector
        self.col2word = self.matrix_maps['col2word']      # vector_to_tokens
        self.leader_row_2_cluster_indices = self.matrix_maps['leader_row_2_cluster_indices']    # cluster
        self.leader_row_2_cluster_ids = self.matrix_maps['leader_row_2_cluster_ids']            # cluster
        self.docID2url = self.matrix_maps['docID2url']                                          # cluster
        self.row2docID = self.matrix_maps['row2docID']

    def load_matrices(self, matrix_names):

        if "leader_document_vector_matrix" in matrix_names:
            # load leader document vector matrix
            ldvm_file_path = file_io.get_path("leader_document_vector_matrix_file_path", [self.output_directory_name])
            self.leader_document_vector_matrix = np.load(ldvm_file_path)

        if "title_document_vector_matrix" in matrix_names:
            # load title document vector matrix
            tdvm_file_path = file_io.get_path("title_document_vector_matrix_file_path", [self.output_directory_name])
            self.title_document_vector_matrix = np.load(tdvm_file_path)

        if "full_document_vector_matrix" in matrix_names:
            # load full document vector matrix
            fdvm_file_path = file_io.get_path("full_document_vector_matrix_file_path", [self.output_directory_name])
            self.full_document_vector_matrix = np.load(fdvm_file_path)

        if "tfidf_matrix" in matrix_names:
            # load tfidf matrix
            tfidf_matrix_file_path = file_io.get_path("tfidf_matrix_file_path", [self.output_directory_name])
            self.tfidf_matrix = np.load(tfidf_matrix_file_path)

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

    def cluster_pruning_search(self, query_vector):

        # TODO: Load in init
        # self.load_matrices(['leader_document_vector_matrix', 'title_document_vector_matrix'])

        # find nearest leader document vector to query vector
        nearest_leader_row = document_vector_operations.ranked_cosine_similarity(query_vector,
                                                                                 self.leader_document_vector_matrix)[0]

        # find the selected clusters document id list and urls
        cluster_indices = np.array(self.leader_row_2_cluster_indices[nearest_leader_row])
        cluster_ids = np.array(self.leader_row_2_cluster_ids[nearest_leader_row])

        #
        # this sorts by title score - does not select leader based on title score
        #

        # add .25 to scores of documents in cluster where the titles contain words in the query
        # the dot product of the query vector and the title vector are greater than 1

        # find title vectors of cluster documents
        title_vectors = self.title_document_vector_matrix[cluster_indices]

        # title_tokens = [self.vector_to_tokens(tv) for tv in title_vectors]
        # print(title_tokens)

        # take the dot product of the query vector against each title vector
        dot_results = np.dot(title_vectors, query_vector)

        # multiply by 0.25 - skip same difference
        # get non zero indices
        # non_zero_indices = np.nonzero(dot_results)

        # sort by max indices and apply this to the cluster ids for ranked results (negative results for reverse order)
        sort_array = np.argsort(-dot_results)
        document_scores = dot_results[sort_array]
        ranked_result_ids = cluster_ids[sort_array]

        return ranked_result_ids, document_scores

    def full_search(self, query_vector):

        # TODO: Load in init
        # self.load_matrices(['title_document_vector_matrix', 'full_document_vector_matrix'])


        #
        # cosine similarity between query vector and all documents
        #

        # cosine similarity between query vector and all documents is the dot product of each row
        full_dot_results = np.dot(self.full_document_vector_matrix, query_vector)

        # normalize scores
        cosine_similarity_scores = np.divide(full_dot_results, query_vector.shape)

        #
        # if any of the query words appear in the title add .25 to its score
        #

        # dot query vector with title_document_vector_matrix
        title_dot_results = np.dot(self.title_document_vector_matrix, query_vector)

        # scale result by 0.25
        title_bonuses = np.multiply(title_dot_results, 0.25)

        # add to the cosine similarity vector for document scores
        scored_documents = np.add(cosine_similarity_scores, title_bonuses)

        #
        # Rank documents
        #

        # get sorted indices
        ranked_result_indices = np.argsort(scored_documents)[::-1]

        # document scores (sort)
        document_scores = scored_documents[ranked_result_indices]

        # remove elements with zeros
        nonzero_indices = np.nonzero(document_scores)
        ranked_result_indices = ranked_result_indices[nonzero_indices]
        document_scores = document_scores[nonzero_indices]

        # get docIDs (don't just add 1)
        row2docID_lambda = lambda row: self.row2docID[row]
        vfunc = np.vectorize(row2docID_lambda)
        ranked_result_ids = vfunc(ranked_result_indices)

        # compute nearest N document vectors to query vector (fast way)
        # nearest_indices = document_vector_operations.ranked_cosine_similarity(query_vector,
        #                                                                         self.full_document_vector_matrix)[:5]

        return ranked_result_ids, document_scores

    def search(self, raw_query, K=6):

        # convert query to vector
        query_vector = self.query_to_vector(raw_query)

        # debug display query tokens
        tokens = self.vector_to_tokens(query_vector)
        logger.debug("query tokens: %s" % str(tokens))

        ranked_result_ids, document_scores = None, None
        if self.search_type == "cluster_pruning":
            ranked_result_ids, document_scores = self.cluster_pruning_search(query_vector)
        if self.search_type == "full_search":
            ranked_result_ids, document_scores = self.full_search(query_vector)

        # only display top k results
        ranked_result_ids = ranked_result_ids[:K]
        document_scores = document_scores[:K]

        #
        # display results
        #

        display_string = "\nUser Query : %s\n\n" % raw_query
        display_string += "\tIndexed Tokens : %s\n\n" % tokens
        display_string += "RESULTS:\n"
        display_string += "-" * 90 + "\n\n"

        display_strings = self.ranked_results_display_strings(ranked_result_ids, document_scores)
        for ds in display_strings:
            display_string += ds
        display_string += "\n\n"
        print(display_string)

    def get_title(self, docID):         # TODO: Merge title databases so outputdir is not needed
        title_path = file_io.get_path('document_title_file_path', [self.output_directory_name, docID])
        with open(title_path) as json_data:
            dtd = json.load(json_data)
            doc_title = dtd['title']
        return doc_title


    def ranked_results_display_strings(self, ranked_result_ids, document_scores):
        ranked_result_urls = [self.docID2url[docID] for docID in ranked_result_ids]
        ranked_result_titles = [self.get_title(docID) for docID in ranked_result_ids]

        urls_titles_and_scores = zip(ranked_result_urls, ranked_result_titles, document_scores)
        title_or_none = lambda title: "NO TITLE" if title == None else title
        format_urls_titles_and_scores = lambda url, title, score: '(SCORE:%s) ' % str(score) + url + '\n' + \
                                                           title_or_none(title) + '\n\n'
        display_strings = [format_urls_titles_and_scores(uts[0], uts[1], uts[2]) for uts in urls_titles_and_scores]

        # add numbers
        for i,ds in enumerate(display_strings):
            display_strings[i] = str(i + 1) + '. ' + ds

        return display_strings
