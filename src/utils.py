import logging
import os
import requests
import json
import sys


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

"""

Incremental Hash ID

    "key": auto incremented integer value
"""

class Incremental_Hash_ID():

    def __init__(self):
        self.table = {}
        self.cur_id = 1

    def add(self, item):
        if item in self.table:
            logger.error("item: %s already in Incremental Hash ID" % item)
        else:
            self.table[item] = self.cur_id
            self.cur_id += 1

    def to_dict(self):
        return self.table

    def load(self, file_path):
        with open(file_path) as json_data:
            #dict.clear(self)
            #self.update(json.load(json_data))
            self.table = json.load(json_data)
            self.cur_id = max(self.table.values()) + 1

    def __getitem__(self, item):
        if item not in self.table:
            logger.error("item: %s not in Incremental Hash ID" % item)
            return None
        return self.table[item]

    def __contains__(self, item):
        if item in self.table:
            return True
        return False

    def __len__(self):
        return len(self.table)

    def __str__(self):
        return str(self.table)


class URL_Frontier():
    """ A queue data structure that ignores added duplicates"""

    def __init__(self):
        self.q = Queue()

    def add(self, data):
        if data not in self.q:
            self.q.add(data)

    def add_list(self, list):
        for d in list:
            self.add(d)

    def remove(self):
        return self.q.remove()

    def __len__(self):
        return len(self.q)

    def __contains__(self, item):
        return item in self.q

    def to_list(self):
        return self.q.to_list()

class Queue():
    """
        This class impliments a minimal queue data structure capable of storing any type of data.

        use:
            q = Queue()         # create queue
            q.add(data)         # add data to tail of queue
            data = q.remove()   # remove and return data from head of queue
    """

    def __init__(self):
        self.head, self.tail = None, None
        self.node = lambda data, next: {'data':data, 'next':next}

    def add(self, data):
        """
        :param data: Data (any type) to add to the tail of queue.
        """
        n = self.node(data, None)

        if self.head == None:
            self.head, self.tail = n, n

        else:
            self.tail['next'] = n
            self.tail = n

    def remove(self):
        """
        :returns: Returns and removes data at the head of the queue or None.
        """
        if self.head is not None:
            n = self.head
            self.head = n['next']
            return n['data']
        return None

    def __len__(self):
        count = 0
        cur = self.head
        while cur is not None:
            cur = cur['next']
            count +=1
        return count

    def __contains__(self, key):
        in_queue = False
        cur = self.head
        while cur is not None:
            if cur['data'] == key:
                in_queue = True
            cur = cur['next']
        return in_queue

    def to_list(self):
        queue_list = []
        cur = self.head
        while cur is not None:
            queue_list.append(cur['data'])
            cur = cur['next']
        return queue_list

"""
    Filter Sites In Bounds and out of bounds
"""

def sub_directory(check_url, parent_url):
    """
    :param check_url: resolved, normalized url string. Child URL in question.
    :param parent_url: resolved, normalized url string. Parent URL in question.
    :return: Boolean return. True if check_url is a sub directory of the parent url.
    """
    common_path = os.path.commonprefix([check_url, parent_url])
    if common_path == parent_url:
        return True
    return False

def filter_sub_directories(list_check_urls, list_parent_urls, filter_if_sub=False):
    """
    :param list_check_urls: list of urls (resolved, normalized url strings) to filter.
    :param list_parent_urls: list of urls (resolved, normalized url strings) to act as bounds, all must be satisfied.
    :param filter_if_sub: Boolean, default False. If set to True, filters directories that are sub directories.
    :return: reduced list of check_urls satisfying condition.
    """
    child_urls = []
    for parent in list_parent_urls:
        get_child = lambda check_url: sub_directory(check_url, parent)
        child_urls += list(filter(get_child, list_check_urls))

    child_urls = set(child_urls)
    if filter_if_sub:
        return list(set(list_check_urls).difference(child_urls))
    return list(child_urls)

"""
    Persistent URL resolver MAP
"""
class URL_Resolver():

    def __init__(self):
        self.url_resolution_map = {}

    def _network_url_resolution(self, url):
        try:
            response = requests.get(url)
            if response.url is None:
                return url
            return response.url
        except:
            return url

    def resolve(self, url):
        if url not in self.url_resolution_map:
            self.url_resolution_map[url] = self._network_url_resolution(url)
        return self.url_resolution_map[url]

    def resolve_list(self, list_of_urls, collapse=False):
        """
        :param list_of_urls:
        :return: list of resolved urls
        """

        for url in list_of_urls:
            if url not in self.url_resolution_map:
                self.url_resolution_map[url] = self._network_url_resolution(url)

        if collapse:
            return list(set([self.url_resolution_map[url] for url in list_of_urls]))
        return [self.url_resolution_map[url] for url in list_of_urls]

    def get_map(self):
        return self.url_resolution_map

    def to_dict(self):
        return self.url_resolution_map

    def save(self, file_path):
        with open(file_path, 'w') as file:
            file.write(json.dumps(self.get_map()))

    def load(self, file_path):
        with open(file_path) as json_data:
            self.url_resolution_map = json.load(json_data)






