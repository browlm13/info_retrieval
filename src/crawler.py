#!/usr/bin/env python

__author__ = "L.J. Brown"
__version__ = "1.0.1"

import hashlib
import requests
from urllib.parse import urljoin
import logging

# my lib
from src import file_parser
from src import webpage_accessor

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Crawler():

    def __init__(self, base_station):
        self.base_station = base_station

    # crawler_id
    def crawl_web_page(self, requested_url, stopwords_file=None):

        # retrieve web page summary
        web_page_summary = webpage_accessor.pull_summary(requested_url, stopwords_file=stopwords_file)

        # report to base station
        index_document = self.base_station.report_web_page_summary(web_page_summary)

        # finish if content has already been indexed (duplicate on content hash)
        if not index_document:
            logger.info("Duplicate Document Found")
            return

        # create document term frequency dictionary
        if web_page_summary['content_type'] in file_parser.acepted_content_types():
            logger.info("Creating Term Frequency Dictionary")
            tfdict = webpage_accessor.pull_summary(requested_url, ('term_frequency_dict'))

            if tfdict is not None:
                if 'term_frequency_dict' in tfdict:
                    if (len(tfdict['term_frequency_dict']) > 0):
                        logger.info("Sending Term Frequency Dictionary")
                        # report to base station
                        self.base_station.report_term_frequency_dictionary(tfdict, web_page_summary['content_hash'])


