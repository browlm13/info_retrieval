import logging
import sys

# my lib
from src import webpage_accessor

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

#
#   Read robots.txt file and return list of normalized disallowed urls
#


def read_robots_dissaloud(site_url):
    forbidden_urls = []

    # get the robots.txt url for the seed url
    robots_url = site_url + "robots.txt"


    # extract the forbidden routes
    response_summary = webpage_accessor.pull_summary(robots_url, ['plain_text'])

    if not 'plain_text' in response_summary:
        logger.error("No Robots.txt Found")
        return []

    robots_string = response_summary['plain_text']

    if robots_string is not None:
        for line in robots_string.split("\n"):
            if line.startswith('Disallow'):
                forbidden = line.split(':')[1].strip()

                if forbidden[0] == '/':
                    forbidden = forbidden[1:]

                if site_url[-1] == '/':
                    site_url = site_url[:-1]

                # add site prefix
                forbidden_urls.append(site_url + '/' + forbidden)

    if len(forbidden_urls) == 0:
        logger.error("No Forbbiden URLs found in Robots.txt")

    return forbidden_urls
