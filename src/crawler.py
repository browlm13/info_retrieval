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
        # web_page_summary = webpage_accessor.pull_summary(requested_url, stopwords_file=stopwords_file)
        included_attributes = ("requested_url", "redirect_history", "status_code", "content_type",
                               "content_hash", "normalized_a_hrefs", 'normalized_img_srcs') #, 'plain_text', 'title')
        web_page_summary = webpage_accessor.pull_summary(requested_url, included_attributes, stopwords_file=stopwords_file)

        # report to base station
        crawler_index_instructions = self.base_station.report_web_page_summary(web_page_summary)

        # break if no content type
        if 'content_type' not in web_page_summary:
            logger.info("No Content Type")
            return

        if crawler_index_instructions['index_html']:
            # return full html if instructed by base station
            if web_page_summary['content_type'] == "text/html":
                page_html = webpage_accessor.pull_summary(requested_url, ('page_html'))
                if page_html is not None:
                    logger.info("Sending Page HTML")
                    # report to base station
                    self.base_station.report_page_html(page_html, web_page_summary['content_hash'])

        if crawler_index_instructions['index_title_and_body']:
            # return text and body plain text json if instructed by base station
            if web_page_summary['content_type'] in file_parser.acepted_content_types():
                title_and_body_dictionary = webpage_accessor.pull_summary(requested_url, ('title', 'plain_text'))
                if title_and_body_dictionary is not None:
                    logger.info("Sending Title And Body")
                    # report to base station
                    self.base_station.report_title_and_body_dictionary(title_and_body_dictionary, web_page_summary['content_hash'])

        if crawler_index_instructions['index_term_frequency_matrix']:
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

