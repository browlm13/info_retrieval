import logging

# external
import PyPDF2   # new
from io import BytesIO  # new
from bs4 import BeautifulSoup

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def acepted_content_types():
    return ["text/html", "text/plain"] #, "application/pdf"]

#
#   Extract Plain Text from response.text and response.headers['content-type']
#


def extract_plain_text( binary_response_content, content_type):
    if content_type[:9] == "text/html":
        return extract_plain_text_html(binary_response_content)
    elif content_type == 'application/pdf':
        return extract_plain_text_pdf(binary_response_content)
    elif content_type == 'text/plain':
        return extract_plain_text_txt(binary_response_content)
    else:
        return "ERROR"


def extract_plain_text_html(binary_response_content):
    soup = BeautifulSoup(binary_response_content, 'html.parser')
    return soup.get_text()


def extract_plain_text_pdf(binary_response_content):
    try:
        pdf = BytesIO(binary_response_content)
        pdfReader = PyPDF2.PdfFileReader(pdf)
        text = ""
        for page_number in range(pdfReader.numPages):
            page = pdfReader.getPage(page_number)
            text += page.extractText()
        return text
    except:
        return "ERROR"


def extract_plain_text_txt(binary_response_content):
    return binary_response_content.decode('UTF-8')
"""
def acepted_content_types():
    return ["text/html", "text/plain"]

#
#   Extract Plain Text from response.text and response.headers['content-type']
#


def extract_plain_text(response_text, content_type):
    if content_type[:9] == "text/html":
        return extract_plain_text_html(response_text)
    elif content_type == 'text/plain':
        return extract_plain_text_txt(response_text)
    else:
        logger.error("incorrect format passed to plain text extractor")
        return None


def extract_plain_text_html(response_text):
    soup = BeautifulSoup(response_text, 'html.parser')
    return soup.get_text()

def extract_plain_text_txt(response_text):
    return response_text
"""

#
#   Extract all <a href="EXTRACT"> from response.text  assuming response.headers['content-type'] == 'text/html'
#

# content type must be text/html


def extract_a_hrefs_list(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    a_hrefs = [l.get('href') for l in soup.find_all('a')]
    return a_hrefs

# content type must be text/html


def extract_img_srcs_list(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    img_srcs = [l.get('src') for l in soup.find_all('img', src=True)]
    return img_srcs

#
#   Extract Title Text from response.text and response.headers['content-type']
#


def extract_title(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup.title.string


#
#   Extract HTML from response.text
#

def extract_html_string(response_text):
    soup = BeautifulSoup(response_text, 'html.parser')
    # listing.prettify()
    return str(soup)
