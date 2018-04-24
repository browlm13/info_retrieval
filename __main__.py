#!/usr/bin/env python
import argparse
import sys
import logging

# my lib
from src import base_station
from src import summary
from src import document_vector_operations as dvo

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

"""
#
# Crawl Site
#

parser = argparse.ArgumentParser( description='Scrape A Website.' )
parser.add_argument('-n', '--number', help='Maximum number of files to index. Will Crawl every page by default.', type=int, default=MAX_URLS_TO_INDEX)
parser.add_argument('-o', '--output', help='Output file name', required=True, type=str)
parser.add_argument('-i', '--input', help='Stopwords File path. Format: one word per line .txt file.', type=str, default=STOPWORDS_FILE)
parser.add_argument('-u', '--url', help='Website to crawl and index.', type=str, default=SEED_URL)
args = parser.parse_args()

# crawl site
bs = base_station.Base_Station()
bs.scrape_website(seed_url=args.url, output_directory=args.output, max_urls_to_index=args.number, stopwords_file=args.input)

# display summary and write term frequency matrix to output file
summary.display_summary(args.output)
"""

dtfm = dvo.get_document_term_frequency_matrix('fmoore')
"""
M, docID2row, word2col = dvo.document_vector_matrix_and_index_dicts(dtfm)
print(word2col)
print(docID2row)
print(M.shape)
import numpy as np
query_vector = np.zeros((1133))
dv12 = M[11,:]
# np.nonzero(dv12)
query_vector[136] = 1
query_vector[137] = 1
query_vector[138] = 1
"""
# testing cosine similarity
# print(dvo.ranked_cosine_similarity(query_vector, M))
dvo.cluster_pruning(dtfm)