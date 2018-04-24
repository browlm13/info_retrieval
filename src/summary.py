import json
import logging

# my lib
from src import utils
from src import file_io
from src import base_station

# external
import glob
import pandas as pd

# logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
summary_logger = logging.getLogger("summary_logger")
summary_logger.propagate = False
fh = logging.FileHandler("output/summary_log.txt", mode='w')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
summary_logger.addHandler(ch)
summary_logger.addHandler(fh)


def display_summary(indexed_directory_name):

    log_info = load_log_file(indexed_directory_name)
    seed_url = log_info['seed_url']

    # log summary
    summary_logger.info("Summary of extracted site information for \"%s\"" % seed_url)

    url_indexer = base_station.URL_Indexer()
    url_indexer.load_url_indexer()
    url_id_map = url_indexer.get_url_id_dict()

    indexed_urls = pd.DataFrame(list(url_id_map.items()),
                 columns=['URL', 'ID'])
    indexed_urls = indexed_urls.sort('ID').set_index('ID')

    # display urls collected
    summary_logger.info("Number new of indexed Urls for site \"%s\" is %s\n" % (seed_url, len(indexed_urls)))
    summary_logger.info(indexed_urls)
    #print("\n\n\n\n\n\nNumber new of indexed Urls for site \"%s\" is %s\n" % (seed_url, len(indexed_urls)))
    #print(indexed_urls)

    # display duplicates
    broken_urls = get_broken_urls(indexed_directory_name)
    summary_logger.info("\nBroken urls found:")
    summary_logger.info(pd.DataFrame(broken_urls, columns=['Broken URL']))
    #print("\nBroken urls found:")
    #print(pd.DataFrame(broken_urls, columns=['Broken URL']))

    # display out of bounds sites
    out_of_bounds_urls = get_out_of_bounds_urls(indexed_directory_name)
    summary_logger.info("\nURLs leading out of seed website: %s \n" % seed_url)
    summary_logger.info(pd.DataFrame(out_of_bounds_urls, columns=["Out of bounds URLs"]))
    #print("\nURLs leading out of seed website: %s " % seed_url)
    #print(pd.DataFrame(out_of_bounds_urls, columns=["Out of bounds URLs"]))

    # display graphic urls
    graphic_urls = get_graphic_urls(indexed_directory_name)
    summary_logger.info("\nGraphic URLs indexed: ")
    summary_logger.info(pd.DataFrame(graphic_urls, columns=["Graphics URLs"]))
    #print("\nGraphic URLs indexed: ")
    #print(pd.DataFrame(graphic_urls, columns=["Graphics URLs"]))

    # display urls pointing to duplicate content
    duplicate_content_urls = get_urls_with_duplicate_content(indexed_directory_name)
    summary_logger.info("\nURLs with duplicate content:")
    summary_logger.info(pd.DataFrame(duplicate_content_urls))
    #print("\nURLs with duplicate content:")
    #, columns=["Duplicate Content URLs"]
    #print(pd.DataFrame(duplicate_content_urls))

    # display document term frequency matrix and most frequently occurring terms
    dtfm = get_document_term_frequency_matrix(indexed_directory_name)
    dtfm_corpus_term_freq = dtfm.sum()
    dtfm_corpus_term_freq.columns = ['term', 'frequency']

    summary_logger.info("\n 20 Most Frequently occurring terms")
    summary_logger.info(dtfm_corpus_term_freq.sort_values(axis=0, ascending=False).head(n=20))
    #summary_logger.info("\n\nDocument Term Frequency Matrix \n")
    #summary_logger.info(dtfm)
    #print("\n\nDocument Term Frequency Matrix \n")
    #print(dtfm)

    # display most frequently occurring terms
    #print("\n 20 Most Frequently occuring terms")
    #print(dtfm_corpus_term_freq.sort_values(axis=0, ascending=False).head(n=20))


def load_log_file(indexed_directory_name):
    log_file_path = file_io.get_path('log_file', [indexed_directory_name])
    if log_file_path is not None:
        with open(log_file_path) as json_data:
            log_info = json.load(json_data)
    return log_info

def load_web_page_summaries(indexed_directory_name):

    web_page_summaries_file_template = file_io.get_template('web_page_access_log_and_metadata_file_path') % (indexed_directory_name, '*')

    web_page_summaries_list = []
    for wpsf_path in glob.glob(web_page_summaries_file_template):
        with open(wpsf_path) as json_data:
            wps = json.load(json_data)
            web_page_summaries_list.append(wps)
    return web_page_summaries_list

def load_document_frequency_dicts(indexed_directory_name):
    document_frequency_dict_file_template = file_io.get_template('document_term_frequency_dictionary_file_path') % (indexed_directory_name, '*')

    document_id_term_frequency_dict = {}
    for dfd_path in glob.glob(document_frequency_dict_file_template):
        with open(dfd_path) as json_data:
            dfd = json.load(json_data)
            doc_id = str(dfd['document_id'])
            doc_term_freq_dict = dfd['term_frequency_dict']
            document_id_term_frequency_dict[doc_id] = doc_term_freq_dict

    return document_id_term_frequency_dict

def get_out_of_bounds_urls(indexed_directory_name):
    wps_list = load_web_page_summaries(indexed_directory_name)
    log_info = load_log_file(indexed_directory_name)
    seed_url = log_info['seed_url']


    all_urls = []
    for wps in wps_list:
        if 'normalized_a_hrefs' in wps:
            all_urls += wps['normalized_a_hrefs']

    url_resolver = utils.URL_Resolver()
    all_urls = url_resolver.resolve_list(all_urls, collapse=True)

    out_of_bounds_urls = utils.filter_sub_directories(all_urls, [seed_url], filter_if_sub=True)
    return out_of_bounds_urls

def get_graphic_urls(indexed_directory_name):
    wps_list = load_web_page_summaries(indexed_directory_name)
    log_info = load_log_file(indexed_directory_name)
    seed_url = log_info['seed_url']


    all_graphic_urls = []
    for wps in wps_list:
        if 'normalized_img_srcs' in wps:
            all_graphic_urls += wps['normalized_img_srcs']

    return list(set(all_graphic_urls))


def get_indexed_urls(indexed_directory_name):
    wps_list = load_web_page_summaries(indexed_directory_name)
    indexed_urls = []
    for wps in wps_list:
        indexed_urls.append(wps['requested_url'])
    return indexed_urls


def get_broken_urls(indexed_directory_name):

    wps_list = load_web_page_summaries(indexed_directory_name)
    broken_urls = []
    for wps in wps_list:
        if wps['status_code'] is not 200:
            broken_urls.append(wps['requested_url'])
    return broken_urls

def get_urls_with_duplicate_content(indexed_directory_name):
    wps_list = load_web_page_summaries(indexed_directory_name)

    content_hash_url_list = {}
    for wps in wps_list:
        if 'content_hash' in wps:
            content_hash = wps['content_hash']
            requested_url = wps['requested_url']
            if content_hash in content_hash_url_list.keys():
                content_hash_url_list[content_hash].append(requested_url)
            else:
                content_hash_url_list[content_hash] = [requested_url]

    urls_with_duplicate_content = []
    for hash, urls in content_hash_url_list.items():
        if len(urls) > 1:
            urls_with_duplicate_content.append(urls)

    return urls_with_duplicate_content


def get_document_term_frequency_matrix(indexed_directory_name, write=True):
    id_tf_dict = load_document_frequency_dicts(indexed_directory_name)

    unique_words = set()
    for doc_id, tf_dict in id_tf_dict.items():
        unique_words = unique_words | set(tf_dict.keys())

    doc_freq_matrix = pd.DataFrame(columns=unique_words)
    for doc_id, tf_dict in id_tf_dict.items():
        terms, freqs = zip(*tf_dict.items())
        df = pd.DataFrame(data=[freqs], columns=terms, index=[doc_id])
        doc_freq_matrix = pd.merge(doc_freq_matrix, df, how='outer')

    doc_freq_matrix = doc_freq_matrix.fillna(value=0)

    # write to csv
    if write:
        logger.info("Writing Document Term Frequency Matrix")
        matrix_file = file_io.get_path("document_term_frequency_matrix_file_path", None, force=True)
        doc_freq_matrix.to_csv(matrix_file)

    return doc_freq_matrix
