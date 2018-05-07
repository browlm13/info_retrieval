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

query = "mary had a little lamb"
qe.search(query, type='full_search')

query = "golfing at smu campus golf club"
qe.search(query, type='full_search')
