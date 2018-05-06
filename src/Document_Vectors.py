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

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))


class DocumentTermFrequencyMatrix():

    def __init__(self):
        # dataFrame rep
        # numpy, dicts rep
        # postings list rep
        pass

    def load_document_term_frequency_dictionaries(self, indexed_directory_name):
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

    def load_from_output_file(self, indexed_directory_name_list):

        # TODO: seperate saved matrices based on indexed_directory_name_list
        #self.indexed_directory_name_list = indexed_directory_name_list
        # check if ANY document term frequency matrix already exists
        document_term_frequency_matrix_file_path = file_io.get_path('document_term_frequency_matrix_file_path', None)
        indexed_directory_name_list_file_path = file_io.get_path('indexed_directory_name_list_file_path', None)

        document_vector_object_already_created = False
        if indexed_directory_name_list_file_path is not None:
            with open(indexed_directory_name_list_file_path) as json_data:
                saved_indexed_directory_name_list_json = json.load(json_data)
            saved_indexed_directory_name_list = saved_indexed_directory_name_list_json['indexed_directory_name_list']
            if saved_indexed_directory_name_list == indexed_directory_name_list:
                if document_term_frequency_matrix_file_path is not None:
                    # load self.df_dtfm from csv and call self.numpy_matrix_and_mappings()
                    document_vector_object_already_created = True
                    self.indexed_directory_name_list_dict = {"indexed_directory_name_list": indexed_directory_name_list}
                    self.df_dtfm = pd.DataFrame.from_csv(document_term_frequency_matrix_file_path)
                    self.numpy_matrix_and_mappings()
        if not document_vector_object_already_created:
            # create and save new stuff
            self.create_from_indexed_directories(indexed_directory_name_list)
            self.save_document_term_frequency_matrix()


    def create_from_indexed_directories(self, indexed_directory_name_list):
        logger.info("Creating Document Term Frequency Matrix from indexed directories: %s" % str(indexed_directory_name_list))

        # NEW
        self.indexed_directory_name_list_dict = {"indexed_directory_name_list" : indexed_directory_name_list}

        id_tf_dict = {}
        for indexed_directory_name in self.indexed_directory_name_list_dict['indexed_directory_name_list']:
            #id_tf_dict = self.load_document_term_frequency_dictionaries(indexed_directory_name)
            id_tf_dict.update(self.load_document_term_frequency_dictionaries(indexed_directory_name))

        logger.info("Finding all unique words...")
        unique_words = set()
        for doc_id, tf_dict in id_tf_dict.items():
            unique_words = unique_words | set(tf_dict.keys())

        logger.info("Filling Pandas DataFrame...")
        doc_freq_matrix = pd.DataFrame(columns=unique_words)  # , index=id_tf_dict.keys())
        doc_freq_matrix.index.name = "docID"

        num_tf_dicts_added = 0
        total_num_tf_dicts = len(id_tf_dict)
        for doc_id, tf_dict in id_tf_dict.items():
            terms, freqs = zip(*tf_dict.items())
            df = pd.DataFrame(data=[freqs], columns=terms, index=[int(doc_id)])
            doc_freq_matrix = pd.concat([doc_freq_matrix, df], join='outer')

            # log completion percentage and loading bar
            process_title = "Document TF index files into Pandas DataFrame"
            num_tf_dicts_added += 1
            completion_percentage = 100 * float(num_tf_dicts_added/total_num_tf_dicts)
            bar_length = 100 # in characters
            divisor = int(100/bar_length)
            loading_bar_string = '[' + '*' * int(completion_percentage/divisor) + \
                                 '_' * (bar_length - int(completion_percentage/divisor)) + ']'
            completion_string = ('Loading %s...\n %s%% Complete.' % (process_title, str(int(completion_percentage)))) + loading_bar_string
            logger.info(completion_string)

        logger.info("Changing NA values to 0's...")
        doc_freq_matrix = doc_freq_matrix.fillna(value=0)

        # sort by docID
        logger.info("Sorting Document Term Frequency Matrix rows by docID ascending...")
        self.df_dtfm = doc_freq_matrix.sort_index()

        # add other representations
        self.numpy_matrix_and_mappings()

    def save_document_term_frequency_matrix(self):

        logger.info("Writing DataFrame Document Term Frequency Matrix")
        df_dtfm_file = file_io.get_path("document_term_frequency_matrix_file_path", None, force=True)
        self.df_dtfm.to_csv(df_dtfm_file)

        logger.info("Writing Numpy Document Term Frequency Matrix")
        np_dtfm_file = file_io.get_path("numpy_document_term_frequency_matrix_file_path", None, force=True)
        #self.df_dtfm.to_csv(np_dtfm_file)
        pd.DataFrame(data=self.np_dtfm).to_csv(np_dtfm_file, index=False)

        logger.info("Writing word to col Map")
        # hack key to string
        data = self.word2col #dict("%s:%s" % (k,v) for k, v in self.word2col.items())
        file_io.save("word2col_file_path", data, None)

        logger.info("Writing DocID to Row Map")
        #hack key to string
        data = dict({str(k):v for k, v in self.docID2row.items()})
        file_io.save("docID2row_file_path", data, None)

        logger.info("Writing DocID to URL Map")
        # hack key to string
        data = dict({str(k):v for k, v in self.docID2url.items()})
        file_io.save("docID2url_file_path", data, None)

        # save indexed directory list
        data = dict(self.indexed_directory_name_list_dict)
        file_io.save("indexed_directory_name_list_file_path", data, None)
        #self.indexed_directory_name_list


    def numpy_matrix_and_mappings(self):
        """
       Makes document_vector_matrix, word2col, docID2row - 2d numpy array matrix with rows as document term frequency vectors, and python
           dictionaries to convert words to column indices and docIDs to row indices
       """
        logger.info("Creating Numpy Representation...")
        self.np_dtfm = self.df_dtfm.as_matrix()

        logger.info("Creating Maps...")
        logger.info("Creating word2col Map...")
        self.word2col = dict(zip(self.df_dtfm.columns.values, range(self.df_dtfm.shape[1])))
        logger.info("Creating docID2row Map...")
        self.docID2row = dict(zip(self.df_dtfm.index.values, range(self.df_dtfm.shape[0])))
        logger.info("Creating docID2url Map...")
        self.docID2url = self.get_docID2url_map()

    def get_docID2url_map(self):

        logger.info("Loading docID2url Map...")
        hash_id_map_file = file_io.get_path("hash_id_map_file", None, force=True)
        with open(hash_id_map_file) as json_data:
            hash_id_map = json.load(json_data)

        hash_url_list_map_file = file_io.get_path("hash_url_list_map_file", None, force=True)
        with open(hash_url_list_map_file) as json_data:
            hash_url_list_map = json.load(json_data)

        # take only first url
        docID_url_map = {hash_id_map[hash]: hash_url_list_map[hash][0] for hash in hash_id_map.keys()}
        return dict(docID_url_map)

    def perform_cluster_pruning_preprocessing(self):

        assert self.df_dtfm is not None
        # logger.info("Performing Cluster ")
        document_vector_operations.save_leader_follower_dictionary(self.df_dtfm)

