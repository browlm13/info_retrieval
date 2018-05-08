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


logger.addHandler(logging.FileHandler("output/output_log.txt"))
# logger.addHandler(logging.StreamHandler(sys.stdout))

#
#   Query Engine
#

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
            self.load_matrices(['title_document_vector_matrix'])
            if weighting_type == "tf":
                self.load_matrices(['leader_document_vector_matrix'])
            if weighting_type == "tfidf":
                self.load_matrices(['tfidf_leader_document_vector_matrix'])
                # self.leader_document_vector_matrix = self.tfidf_leader_document_vector_matrix
                # self.leader_row_2_cluster_indices = self.tfidf_leader_row_2_cluster_indices
                # self.leader_row_2_cluster_ids = self.tfidf_leader_row_2_cluster_ids
                # TODO: Implement tfidf option in cluster pruning search

        if search_type == "full_search":
            self.load_matrices(['title_document_vector_matrix'])
            if weighting_type == "tf":
                self.load_matrices(['full_document_vector_matrix'])
            if weighting_type == "tfidf":
                self.load_matrices(['tfidf_matrix'])
                # self.full_document_vector_matrix = self.tfidf_matrix
                # TODO: Implement tfidf option in cluster pruning search

    def display_clustering_info(self, write=False, method="using_kmeans"):

        # tf cluster information
        id_clusters = list(self.leader_row_2_cluster_ids.values())
        indices_clusters = list(self.leader_row_2_cluster_indices.values())
        info_string = "term frequency clustering information: (cluster leaders chosen %s)" % method
        info_string += self.clusters_info_string(id_clusters, indices_clusters)

        # tfidf cluster information
        id_clusters = list(self.tfidf_leader_row_2_cluster_ids.values())
        indices_clusters = list(self.tfidf_leader_row_2_cluster_indices.values())
        info_string += "\n\ntf-idf clustering information: (cluster leaders chosen %s)" % method
        info_string += self.clusters_info_string(id_clusters, indices_clusters)

        print(info_string)
        if write==True:
            # tmp write results to file
            with open("clustering_info.txt", "w") as myfile:
                myfile.write(info_string)



    def clusters_info_string(self, id_clusters, indices_clusters):

        # by ids
        # id_clusters = list(self.leader_row_2_cluster_ids.values())
        id_cluster_matrix = np.array(id_clusters)
        id_leaders = id_cluster_matrix[:,0]
        id_followers_matrix = id_cluster_matrix[:,1:]

        # by indices
        # indices_clusters = list(self.leader_row_2_cluster_indices.values())
        indices_cluster_matrix = np.array(indices_clusters)

        # indices_leaders = indices_cluster_matrix[:,0]
        indices_follower_matrix = indices_cluster_matrix[:,1:]

        # compute leader follower score for each element in row 0 for each row
        self.load_matrices(['full_document_vector_matrix'])
        self.load_matrices(['leader_document_vector_matrix'])
        leader_follower_pair_scores_matrix = np.empty(shape=(len(indices_clusters), len(indices_clusters[0]) -1))
        for i in range(len(indices_clusters)):
            followers_i_document_vectors = self.full_document_vector_matrix[indices_follower_matrix[i,:]]
            leader_i_document_vector = self.leader_document_vector_matrix[i]

            # compute cosine similarity between leader and each follower / row in followers_i_document_vectors
            leader_followers_i_dot_results = np.dot(followers_i_document_vectors, leader_i_document_vector)

            # normalize scores
            leader_follower_i_cosine_similarity_scores = \
                np.divide(leader_followers_i_dot_results, leader_i_document_vector.shape)

            # set row of leader_follower_pair_scores_matrix
            leader_follower_pair_scores_matrix[i] = leader_follower_i_cosine_similarity_scores

        # display leader docID , follower docID and pair scores for each follower of leader - for each leader
        cluster_pruning_info_string = ''
        for i in range(len(id_clusters)):
            i_leader_id = id_leaders[i]
            i_follower_ids = id_followers_matrix[i]
            i_leader_follower_scores = leader_follower_pair_scores_matrix[i]
            i_follower_id_score_tuple_list = zip(i_follower_ids, i_leader_follower_scores)
            cluster_pruning_info_string += "\ncluster %s:" % (i + 1)
            cluster_pruning_info_string += "\n\tleader docID: %s" % i_leader_id
            cluster_pruning_info_string += "\n\tfollower docIDs and leader/follower scores:"
            for id, score in i_follower_id_score_tuple_list:
                cluster_pruning_info_string += "\n\t\tdocID: %s, score: %s" % (id, score)

        return cluster_pruning_info_string

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
        self.tfidf_leader_row_2_cluster_indices = self.matrix_maps['tfidf_leader_row_2_cluster_indices']# tfidf cluster
        self.tfidf_leader_row_2_cluster_ids = self.matrix_maps['tfidf_leader_row_2_cluster_ids']        # tfidf cluster
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

        if "tfidf_leader_document_vector_matrix" in matrix_names:
            # load leader document vector matrix
            tfidf_ldvm_file_path = file_io.get_path("tfidf_leader_document_vector_matrix_file_path",
                                              [self.output_directory_name])
            self.tfidf_leader_document_vector_matrix = np.load(tfidf_ldvm_file_path)

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


        if self.weighting_type == 'tf':

            # find nearest leader document vector to query vector
            nearest_leader_row = document_vector_operations.ranked_cosine_similarity(query_vector,
                                                                                     self.leader_document_vector_matrix)[0]

            # find the selected clusters document id list and urls
            cluster_indices = np.array(self.leader_row_2_cluster_indices[nearest_leader_row])
            cluster_ids = np.array(self.leader_row_2_cluster_ids[nearest_leader_row])
        if self.weighting_type == 'tfidf':
            # find nearest leader document vector to query vector
            nearest_leader_row = document_vector_operations.ranked_cosine_similarity(query_vector,
                                                                        self.tfidf_leader_document_vector_matrix)[0]

            # find the selected clusters document id list and urls
            cluster_indices = np.array(self.tfidf_leader_row_2_cluster_indices[nearest_leader_row])
            cluster_ids = np.array(self.tfidf_leader_row_2_cluster_ids[nearest_leader_row])


        #
        # give each doc a base score of 1
        #
        base_scores = np.ones(cluster_ids.shape)

        #
        # if any of the query words appear in the title add .25 to its score
        #
        # this sorts by title score - does not select leader based on title score

        # find title vectors of cluster documents
        title_vectors = self.title_document_vector_matrix[cluster_indices]

        # title_tokens = [self.vector_to_tokens(tv) for tv in title_vectors]
        # print(title_tokens)

        # take the dot product of the query vector against each title vector
        title_dot_results = np.dot(title_vectors, query_vector)

        # scale result by 0.25
        title_bonuses = np.multiply(title_dot_results, 0.25)

        # add to the cosine similarity vector for document scores
        scored_documents = np.add(base_scores, title_bonuses)

        #
        # sort based on score from title
        #

        # sort by max indices and apply this to the cluster ids for ranked results (negative results for reverse order)
        sort_array = np.argsort(-scored_documents)
        document_scores = scored_documents[sort_array]
        ranked_result_ids = cluster_ids[sort_array]

        """
        # count nonzero scores
        num_results = np.count_nonzero(document_scores)
        if num_results < 1:
            logger.debug("No Results Found")
            return np.zeros(0), np.zeros(0)
        
        return ranked_result_ids[:num_results], document_scores[:num_results]
        """

        if not document_scores.any():
            logger.debug("No Results Found")
            return np.zeros(0), np.zeros(0)

        return ranked_result_ids, document_scores

    def full_search(self, query_vector):

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

        # count nonzero scores
        num_results = np.count_nonzero(document_scores)
        if num_results < 1:
            logger.debug("No Results Found")
            return np.zeros(0), np.zeros(0)

        # remove elements with zeros
        nonzero_indices = np.nonzero(document_scores)
        ranked_result_indices = ranked_result_indices[nonzero_indices]
        document_scores = document_scores[nonzero_indices]

        # get docIDs (don't just add 1)
        row2docID_lambda = lambda row: self.row2docID[row]
        vfunc = np.vectorize(row2docID_lambda)
        ranked_result_ids = vfunc(ranked_result_indices)

        # compute nearest N document vectors to query vector (the fast way -- does not give scores)
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

        #
        # display results
        #

        display_string = "\nUser Query : %s\n\n" % raw_query
        display_string += "\tIndexed Tokens : %s\n\n" % tokens
        display_string += "\tSearch Type: %s, Weighting Type: %s\n\n" % (self.search_type, self.weighting_type)
        display_string += "RESULTS:\n"
        display_string += "-" * 90 + "\n\n"

        # results found
        num_results = int(np.count_nonzero(document_scores))
        if (self.search_type == "cluster_pruning") and (num_results != 0):
            # display entire cluster
            pass

        elif num_results >= K:
            # only display top k results
            ranked_result_ids = ranked_result_ids[:K]
            document_scores = document_scores[:K]

        else:
            logger.info("Less than %s results found" % str(K))

            if num_results < int(K/2.0):

                logger.info("Less than K/2 results found")
                # raise Exception('Less than K/2 results found')
                # run query expansion

            if num_results == 0:
                display_string += "No Results Found."

        if num_results > 0:
            display_strings = self.ranked_results_display_strings(ranked_result_ids, document_scores)
            for ds in display_strings:
                display_string += ds

        display_string += "\n\n"
        print(display_string)

        # tmp write results to file
        with open("results.txt", "a") as myfile:
            myfile.write(display_string)

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
        format_urls_titles_and_scores = lambda url, title, score: 'URL: %s,\nTITLE: %s,\nSCORE: %s\n\n' \
                                                                  % (url, title_or_none(title), str(score))
        display_strings = [format_urls_titles_and_scores(uts[0], uts[1], uts[2]) for uts in urls_titles_and_scores]

        # add numbers
        for i,ds in enumerate(display_strings):
            display_strings[i] = str(i + 1) + '. ' + ds

        return display_strings
