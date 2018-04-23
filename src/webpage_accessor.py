#!/usr/bin/env python

__author__ = "L.J. Brown"
__version__ = "1.0.1"

import hashlib
import requests
from urllib.parse import urljoin
import logging
import sys

# mylib
from src import file_parser
from src import text_processing

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

import hashlib
import requests
import logging

# my lib
from src import file_parser
from src import text_processing

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pull_summary(requested_url, included_attributes=("requested_url", "redirect_history", "status_code", "content_type",
                                                     "content_hash", "normalized_a_hrefs", 'normalized_img_srcs'),
                 stopwords_file=None):
    """ access a given url and return a python dictionary of page data. """

    response_summary = {
        'requested_url': requested_url,
        'status_code': 404
    }

    # set 'requested_url' value

    try:

        # make request
        response = requests.get(requested_url)

        # set "status_code" value
        response_summary['status_code'] = response.status_code

        # log status code
        logger.info("Response Status Code: %d" % response.status_code)

        # continue if status is 200
        if response.status_code == 200:

            # set 'content_hash' value
            response_summary['content_hash'] = str(hashlib.md5(str.encode(response.text)).hexdigest())

            # set 'redirect_history'  value
            response_summary['redirect_history'] = []
            for redirect in response.history:
                response_summary['redirect_history'].append(redirect.url)

            # set 'content_type' value
            if 'content-type' in response.headers:
                response_summary['content_type'] = response.headers['content-type']

            # set 'binary_response_content' value
            if 'binary_response_content' in included_attributes:
                response_summary['binary_response_content'] = response.content

            # set 'plain_text' value
            if 'plain_text' in included_attributes:
                response_summary['plain_text'] = None
                if response_summary['content_type'] in file_parser.acepted_content_types():
                    response_summary['plain_text'] = file_parser.extract_plain_text(response.text,
                                                                                    response_summary['content_type'])

            # set 'tokens' value
            if 'tokens' in included_attributes:
                if response_summary['content_type'].split(';')[0] in file_parser.acepted_content_types():
                    plain_text = file_parser.extract_plain_text(response.text, response_summary['content_type'])
                    response_summary['tokens'] = text_processing.plain_text_to_tokens(plain_text)

            if 'term_frequency_dict' in included_attributes:
                if response_summary['content_type'].split(';')[0] in file_parser.acepted_content_types():
                    plain_text = file_parser.extract_plain_text(response.text, response_summary['content_type'])
                    tokens = text_processing.plain_text_to_tokens(plain_text, stopwords_file)
                    response_summary['term_frequency_dict'] = text_processing.word_frequency_dict(tokens)

            # if type "text/html" - read links
            if ('normalized_a_hrefs' in included_attributes) or ('normalized_img_srcs' in included_attributes):
                if response.headers['content-type'][:9] == "text/html":
                    # Note: base_url is requested_url

                    # set 'normalized_a_hrefs'
                    response_summary['normalized_a_hrefs'] = text_processing.normalize_urls(requested_url,
                                                                            file_parser.extract_a_hrefs_list(
                                                                                response.text))

                    # set 'normalized_img_srcs'
                    response_summary['normalized_img_srcs'] = text_processing.normalize_urls(requested_url,
                                                                             file_parser.extract_img_srcs_list(
                                                                                 response.text))

    except:
        logger.warning("Requested Page: %s, Failed to read." % response_summary['requested_url'])

    # filter attributes not in included_attributes tuple parameter
    response_summary = {k: v for k, v in response_summary.items() if k in included_attributes}

    return response_summary
