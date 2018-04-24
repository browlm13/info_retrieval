import logging
import sys

# my lib
from src import utils
from src import file_io
from src import crawler
from src import robot_parser

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

# TODO: add url_id : content hash map
# TODO: add url graph data structure for connected resolved links

class URL_Indexer():

    def __init__(self):
        self.url_resolver = utils.URL_Resolver()
        self.url_id_index = utils.Incremental_Hash_ID()

    def get_url_id_dict(self):
        return self.url_id_index.to_dict()

    def url_in_index(self, url):
        """ Resolves passed in url and checks if it is in index. returns boolean."""
        resolved_url = self.url_resolver.resolve(url)
        if resolved_url in self.url_id_index:
            return True
        return False

    def filter_in_index(self, url_list):
        filtered_urls = []
        for url in url_list:
            if not self.url_in_index(url):
                filtered_urls.append(url)
        return filtered_urls

    def resolve_url(self, url):
        return self.url_resolver.resolve(url)

    def resolve_url_list(self, url_list):
        return self.url_resolver.resolve_list(url_list)

    def add_web_page_summary(self, web_page_summary, output_directory_name):
        """ Resolves web page links, Indexes and writes web page summary only if resolved requested url is not in index."""

        # resolved requested url
        requested_url = web_page_summary['requested_url']
        resolved_requested_url = self.url_resolver.resolve(requested_url)

        # check if resolved requested url is in index. if it is, return
        if resolved_requested_url in self.url_id_index:
            return

        logger.info("Adding new URL to index: %s" % resolved_requested_url)

        # add new url to index
        self.url_id_index.add(resolved_requested_url)

        # if not in index, resolve all web page links and write to file

        # and if has attributes
        resolved_normalized_a_hrefs = []
        if 'normalized_a_hrefs' in web_page_summary:
            resolved_normalized_a_hrefs = self.url_resolver.resolve_list(web_page_summary['normalized_a_hrefs'])
        resolved_normalized_img_srcs = []
        if 'normalized_img_srcs' in web_page_summary:
            resolved_normalized_img_srcs = self.url_resolver.resolve_list(web_page_summary['normalized_img_srcs'])

        # add web_page_summary  resolved links
        # and add the additional url_id key value pair before writing to file
        written_web_page_summary = web_page_summary.copy()
        written_web_page_summary['url_id'] = self.url_id_index[resolved_requested_url]
        written_web_page_summary['resolved_requested_url'] = resolved_requested_url
        written_web_page_summary['resolved_normalized_a_hrefs'] = resolved_normalized_a_hrefs
        written_web_page_summary['resolved_normalized_img_srcs'] = resolved_normalized_img_srcs

        # write file
        logger.info("Saving response summary")
        file_io.save('web_page_access_log_and_metadata_file_path', written_web_page_summary,
                     [output_directory_name, written_web_page_summary['url_id']])


    def save_url_indexer(self):
        # write file
        file_io.save('url_id_map_file', self.url_id_index.to_dict(), None)
        file_io.save('resolved_url_map_file', self.url_resolver.to_dict(), None)

    def load_url_indexer(self):

        # load url_id_index
        url_indexer_file_path = file_io.get_path('url_id_map_file', None)
        if url_indexer_file_path is not None:
            self.url_id_index.load(url_indexer_file_path)

        # load url_resolver
        url_resolver_file_path = file_io.get_path('resolved_url_map_file', None)
        if url_resolver_file_path is not None:
            self.url_resolver.load(url_resolver_file_path)

    def __len__(self):
        return len(self.url_id_index)


class Document_Indexer():

    def __init__(self):
        self.hash_id_index = utils.Incremental_Hash_ID()

    def document_in_index(self, content_hash):
        """ checks if document/content hash it is in index. returns boolean."""

        if content_hash in self.hash_id_index:
            return True
        return False

    """
    # new
    def content_hash(self, document_id):
        id_hash_index = {v:k for k,v in self.hash_id_index.to_dict().items()}
        assert document_id in id_hash_index
        return id_hash_index[document_id]

    # new
    def document_id(self, content_hash):
        assert self.document_in_index(content_hash)
        return self.hash_id_index[content_hash]

    # new
    def add_document(self, content_hash):
        self.hash_id_index.add(content_hash)
    """

    def save_term_frequency_dictionary(self, term_frequency_dictionary, content_hash, output_directory_name):
        # add new document hash to index
        self.hash_id_index.add(content_hash)
        document_id = self.hash_id_index[content_hash]

        # add doc id and hash to term frequency dictionary
        term_frequency_dictionary['document_id'] = document_id
        term_frequency_dictionary['content_hash'] = content_hash

        logger.info("Saving Document Term Frequency Dictonary ID: %d" % document_id)
        # write file
        file_io.save('document_term_frequency_dictionary_file_path', term_frequency_dictionary,
                     [output_directory_name, document_id])

    # new
    def save_document_html_dictionary(self, document_html, content_hash, output_directory_name):
        # add new document hash to index
        self.hash_id_index.add(content_hash)
        document_id = self.hash_id_index[content_hash]

        # create page html json with meta data
        document_html_dictionary = {
            'page_html' : document_html,
            'document_id' : document_id,
            'content_hash' : content_hash
        }

        logger.info("Saving Page HTML, ID: %d" % document_id)
        # write file
        file_io.save('document_html_file_path', document_html_dictionary,
                     [output_directory_name, document_id])

    def save_document_title_dictionary(self, document_title, content_hash, output_directory_name):
        # add new document hash to index
        self.hash_id_index.add(content_hash)
        document_id = self.hash_id_index[content_hash]

        # create page html json with meta data
        document_title_dictionary = {
            'title' : document_title['title'],
            'document_id' : document_id,
            'content_hash' : content_hash
        }
        logger.info("Saving Document Title, ID: %d" % document_id)
        # write file
        file_io.save('document_title_file_path', document_title_dictionary,
                     [output_directory_name, document_id])

    def save_document_plain_text_dictionary(self, document_plain_text, content_hash, output_directory_name):
        # add new document hash to index
        self.hash_id_index.add(content_hash)
        document_id = self.hash_id_index[content_hash]

        # create page html json with meta data
        document_plain_text_dictionary = {
            'plain_text' : document_plain_text['plain_text'],
            'document_id' : document_id,
            'content_hash' : content_hash
        }

        logger.info("Saving Document Plain Text, ID: %d" % document_id)
        # write file
        file_io.save('document_plain_text_file_path', document_plain_text_dictionary,
                     [output_directory_name, document_id])


    def save_document_indexer(self):
        # write file
        file_io.save('doc_hash_id_map_file', self.hash_id_index.to_dict(), None)

    def load_document_indexer(self):
        document_indexer_file_path = file_io.get_path('doc_hash_id_map_file', None)
        if document_indexer_file_path is not None:
            self.hash_id_index.load(document_indexer_file_path)


class Base_Station():

    def __init__(self, index_document_html=False, index_document_title=False, index_document_plain_text=False, index_document_term_frequency_dictionary=True):
        self.url_indexer = URL_Indexer()
        self.document_indexer = Document_Indexer()

        # content_hash : url list - map

        # load indexers
        self.load_indexes()

        # elements to index
        self.crawler_index_instructions = {
            "index_document_html": index_document_html,
            "index_document_title": index_document_title,
            "index_document_plain_text": index_document_plain_text,
            "index_document_term_frequency_dictionary": index_document_term_frequency_dictionary
        }

    def update_frontier(self, url_list):
        # filter
        # resolve
        filtered_urls = self.url_indexer.resolve_url_list(url_list)

        # filter urls already in index
        filtered_urls = self.url_indexer.filter_in_index(filtered_urls)

        # filter links not within site bounds
        filtered_urls = utils.filter_sub_directories(filtered_urls, [self.seed_url])

        # filter robots
        filtered_urls = utils.filter_sub_directories(filtered_urls, self.forbidden_urls, filter_if_sub=True)

        # add to url frontier
        self.url_frontier.add_list(filtered_urls)

    def continue_indexing(self):

        # stop if url queue is empty
        if len(self.url_frontier) == 0:
            return False

        # stop if max_urls_to_index param has been reached
        if self.max_urls_to_index is not None:
            return len(self.url_indexer) < self.max_urls_to_index

        return True

    def scrape_website(self, seed_url, output_directory, max_urls_to_index=None, stopwords_file=None):
        self.seed_url = self.url_indexer.resolve_url(seed_url)
        self.output_directory_name = output_directory
        self.max_urls_to_index = max_urls_to_index
        self.stopwords_file = stopwords_file

        # read robots
        self.forbidden_urls = self.url_indexer.resolve_url_list(robot_parser.read_robots_dissaloud(self.seed_url))

        # make queue
        self.url_frontier = utils.URL_Frontier()

        # add seed to url frontier
        self.update_frontier([self.seed_url])

        # log
        self.write_log_file()

        # create crawler
        c = crawler.Crawler(self)

        # crawl size
        while self.continue_indexing():
            next_url = self.url_frontier.remove()
            logger.info("Crawling: %s" % next_url)
            logger.info("Number of sites in index: %d" % len(self.url_indexer))
            c.crawl_web_page(next_url, self.stopwords_file)

            # save maps periodically
            self.save_indexes()

        # save maps
        self.save_indexes()

    def report_web_page_summary(self, web_page_summary):
        """ Recieves a web page summary dictonary from a crawler. Checks the content hash for content already indexed. Returns True if Not yet indexed"""

        self.url_indexer.add_web_page_summary(web_page_summary, self.output_directory_name)

        # update_frontier
        if 'normalized_a_hrefs' in web_page_summary:
            self.update_frontier(web_page_summary['normalized_a_hrefs'])

        # checks if document has been indexed
        document_in_index = False
        if 'content_hash' in web_page_summary:
            content_hash = web_page_summary['content_hash']
            document_in_index = self.document_indexer.document_in_index(content_hash)  # continue indexing...

        """
        # checks if document has been indexed
        document_in_index = False
        if 'content_hash' in web_page_summary:
            content_hash = web_page_summary['content_hash']
            document_in_index = self.document_indexer.document_in_index(content_hash)  # continue indexing...

            # indent added
            # add document to document indexer
            self.document_indexer.add_document(content_hash)
            document_id = self.document_indexer.document_id(content_hash)

            # add document id field to url indexer web_page summary
            web_page_summary['document_id'] = document_id

            # add to url indexer
            self.url_indexer.add_web_page_summary(web_page_summary, self.output_directory_name)

            # update_frontier
            if 'normalized_a_hrefs' in web_page_summary:
                self.update_frontier(web_page_summary['normalized_a_hrefs'])
        """

        ### TODO: add content_hash : [URL list]

        # elements to index
        # if document has been indexed, negate all other elements
        crawler_index_instructions = self.crawler_index_instructions
        if document_in_index:
            crawler_index_instructions = {k:False for k, v in self.crawler_index_instructions.items()}

        # return crawler index instructions
        return crawler_index_instructions

    def report_document_term_frequency_dictionary(self, term_frequency_dictionary, content_hash):
        """ Recieves a web page summary dictonary from a crawler.
        Checks the content hash for contnet already indexed."""
        self.document_indexer.save_term_frequency_dictionary(term_frequency_dictionary, content_hash,
                                                             self.output_directory_name)

    # new
    def report_document_html(self, document_html, content_hash):
        """ Receives web page html string from a crawler. Checked the content hash for content already indexed."""
        self.document_indexer.save_document_html_dictionary(document_html, content_hash,
                                                             self.output_directory_name)

    def report_document_title(self, document_title, content_hash):
        """ Receives web page html string from a crawler. Checked the content hash for content already indexed."""
        self.document_indexer.save_document_title_dictionary(document_title, content_hash, self.output_directory_name)


    def report_document_plain_text(self, document_plain_text, content_hash):
        """ Receives web page html string from a crawler. Checked the content hash for content already indexed."""
        self.document_indexer.save_document_plain_text_dictionary(document_plain_text, content_hash, self.output_directory_name)

    def save_indexes(self):
        logger.info("Saving Index Files")
        # add log
        self.document_indexer.save_document_indexer()
        self.url_indexer.save_url_indexer()

    def load_indexes(self):
        logger.info("Loading Index Files")
        self.url_indexer.load_url_indexer()
        self.document_indexer.load_document_indexer()

    def write_log_file(self):
        self.log_info_dict = {}
        self.log_info_dict['seed_url'] = self.seed_url
        self.log_info_dict['max_urls_to_index'] = self.max_urls_to_index
        self.log_info_dict['robots_dissaloud_path'] = self.forbidden_urls
        self.log_info_dict['indexed_elements'] = self.crawler_index_instructions

        # write file
        file_io.save('log_file', self.log_info_dict, [self.output_directory_name])