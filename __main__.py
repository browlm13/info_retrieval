#!/usr/bin/env python
import argparse
import sys
import logging

# my lib
from src import base_station
from src import matrices_and_maps
from src import query_engine
from src import summary

__author__ = 'LJ Brown'
__version__ = "2.0.1"

"""
                        Search Engine / Web Crawler
                             Command Line Tool
"""
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

# Default Parameters
SEED_URL = "http://lyle.smu.edu/~fmoore/"
MAX_URLS_TO_INDEX = None
STOPWORDS_FILE = "stopwords.txt"


#
#   Preprocessing
#

parser = argparse.ArgumentParser( description='Scrape A Website.' )
parser.add_argument('-n', '--number', help='Maximum number of files to index. Will Crawl every page by default.', type=int, default=MAX_URLS_TO_INDEX)
parser.add_argument('-o', '--output', help='Output file name', required=True, type=str)
parser.add_argument('-i', '--input', help='Stopwords File path. Format: one word per line .txt file.', type=str, default=STOPWORDS_FILE)
parser.add_argument('-u', '--url', help='Website to crawl and index.', type=str, default=SEED_URL)
args = parser.parse_args()



#
#   Crawl Site
#

# bs = base_station.Base_Station(index_document_html=False, index_document_title=True, index_document_plain_text=True, index_document_term_frequency_dictionary=True)
# bs.scrape_website(seed_url=args.url, output_directory=args.output, max_urls_to_index=args.number, stopwords_file=args.input)

#
#   Build Matrices and Maps
#

# matrices_and_maps.build_matrices_and_maps([args.output])

#
#
#   Run Search Engine
#
#

qe = query_engine.QueryEngine(args.output, search_type="full_search", weighting_type="tfidf")

# qe.display_clustering_info(write=True)
qe.display_clustering_info()

# required queries
"""
query = "moore smu"
qe.search(query)

query = "Bob Ewell where Scout"
qe.search(query)

query = "three year story"
qe.search(query)

query = "Atticus to defend Maycomb"
qe.search(query)

query = "hocuspocus thisworks"
qe.search(query)
"""

# make program loop

welcome_string = "\n\n"
welcome_string += '=' * 90
welcome_string += '\n\n'
welcome_string += '\tSearch Engine (preprocessing done on \"http://lyle.smu.edu/~fmoore/\")\n\n'
welcome_string += '=' * 90

while True:
    print(welcome_string)
    print("\tSettings: Search Type: %s, Weighting Type: %s" % (qe.search_type, qe.weighting_type))
    print("Type Query then press \'Enter\'. Search single word \'STOP\' (case sensitive) to terminate.\n")
    query = input("[Search]: ")

    # terminate condition
    if query == "STOP":
        print("Stopping Program.")
        sys.exit()

    # search for user query
    qe.search(query)



"""
# qe = query_engine.QueryEngine(args.output, search_type="full_search")
qe = query_engine.QueryEngine(args.output, search_type="full_search", weighting_type="tfidf")
# qe = query_engine.QueryEngine(args.output, search_type="cluster_pruning")

query = "moore smu"
qe.search(query)

query = "Bob Ewell where Scout"
qe.search(query)

query = "three year story"
qe.search(query)

query = "Atticus to defend Maycomb"
qe.search(query)

query = "hocuspocus thisworks"
qe.search(query)

"SMU CSE 5337/7337 Spring 2018 Schedule" # https://s2.smu.edu/~fmoore/schedule.htm
"Freeman Moore - SMU Spring 2018" # https://s2.smu.edu/~fmoore/index_duplicate.htm
"Freeman Moore - SMU Spring 2017" # https://s2.smu.edu/~fmoore/index-final.htm
"""