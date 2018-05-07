#!/usr/bin/env python
import argparse
import sys
import logging

# my lib
from src import Document_Vectors
from src import document_vector_operations as dvo
from src import base_station
from src import matrices_and_maps
from src import query_engine

__author__ = 'LJ Brown'
__version__ = "2.0.1"

"""
                        Search Engine / Web Crawler
                             Command Line Tool
"""

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

# Default Parameters
SEED_URL = "http://lyle.smu.edu/~fmoore/"
MAX_URLS_TO_INDEX = None
STOPWORDS_FILE = "stopwords.txt"


#
# Crawl Site
#

parser = argparse.ArgumentParser( description='Scrape A Website.' )
parser.add_argument('-n', '--number', help='Maximum number of files to index. Will Crawl every page by default.', type=int, default=MAX_URLS_TO_INDEX)
parser.add_argument('-o', '--output', help='Output file name', required=True, type=str)
parser.add_argument('-i', '--input', help='Stopwords File path. Format: one word per line .txt file.', type=str, default=STOPWORDS_FILE)
parser.add_argument('-u', '--url', help='Website to crawl and index.', type=str, default=SEED_URL)
args = parser.parse_args()

# TODO: add options for saving, document frequency matrix, html, and plain text

"""
# crawl site
bs = base_station.Base_Station(index_document_html=False, index_document_title=True, index_document_plain_text=True, index_document_term_frequency_dictionary=True)
bs.scrape_website(seed_url=args.url, output_directory=args.output, max_urls_to_index=args.number, stopwords_file=args.input)
"""
# display summary and write term frequency matrix to output file
### summary.display_summary(args.output) # builds shitty matrix

#
#   Build Matrices and Maps
#

# matrices_and_maps.build_matrices_and_maps([args.output])

qe = query_engine.QueryEngine(args.output)

query = "SMU CSE 5337/7337 Spring 2018 Schedule" # https://s2.smu.edu/~fmoore/schedule.htm
qe.search(query, type='full_search')

query = "Freeman Moore - SMU Spring 2018" # https://s2.smu.edu/~fmoore/index_duplicate.htm
qe.search(query, type='full_search')

query = "Freeman Moore - SMU Spring 2017" # https://s2.smu.edu/~fmoore/index-final.htm
qe.search(query, type='full_search')

#
# Build Document Term Frequency Matrix
#

# dtfm = Document_Vectors.DocumentTermFrequencyMatrix()
# dtfm.load_from_output_file([args.output])

#
# Perform Cluster Pruning and save Leader Follower Dictonary
#
#dtfm.perform_cluster_pruning_preprocessing()


# test load leader follower dict
#leader_follower_dict = dvo.load_leader_follower_dictionary()



#dtfm = dvo.get_document_term_frequency_matrix(args.output)
#print(dtfm)
#M, docID2row, word2col = dvo.document_vector_matrix_and_index_dicts(dtfm)
#print(word2col)
#print(docID2row)
#print(M.shape)
#import numpy as np

#dv12 = M[11,:]



"""
# testing query to vector and back
raw_query = "advanc amid attack"
query_vector = dvo.query_to_vector(raw_query, dtfm)
token_list = dvo.vector_to_tokens(query_vector, dtfm)
print(token_list)
"""
"""
# get docID_url_map
docID_url_map = dvo.get_docID2url_map()

dtfm = dvo.get_document_term_frequency_matrix(args.output)
M, docID2row, word2col = dvo.document_vector_matrix_and_index_dicts(dtfm)


# cluster pruning
import numpy as np
# load saved leader follower dictonary
leader_follower_docID_dict = dvo.load_leader_follower_dictionary()
# get leader indices as np array
leader_IDs = list(leader_follower_docID_dict.keys())
leader_indices = np.array([docID2row[lid] for lid in leader_IDs])

# get leader vectors as matrix
leader_M = M[leader_indices]



# query to vector
# raw_query = "advanc amid attack"
raw_query = "buildingthree buildingtwo buildingone"
# raw_query = "Mary's lambs are little"
# raw_query = "player"
query_vector = dvo.query_to_vector_slow(raw_query)


# test website retreival without cluster pruning
similar_indices = dvo.ranked_cosine_similarity(query_vector, dtfm)
row2docID = {v : k for k,v in docID2row.items()}
similar_ids = [row2docID[idx] for idx in similar_indices]
url_list = [docID_url_map[rid] for rid in similar_ids]
print(url_list)


# get max leader index based on cosine similarity with query vector
most_similar_leader_index = dvo.ranked_cosine_similarity(query_vector, leader_M)[0]

# get max leader id
most_similar_leader_id = leader_IDs[most_similar_leader_index]

# get docID's to return as list by adding add follower ids
follower_IDs = leader_follower_docID_dict[most_similar_leader_id]
result_ids = [most_similar_leader_id] + follower_IDs

# get result urls
url_list = [docID_url_map[rid] for rid in result_ids]
print(url_list)

"""