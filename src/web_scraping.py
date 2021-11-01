# common installs
# pip install bs4

# common imports
from bs4 import BeautifulSoup, Comment
import requests
from googlesearch import search
import html2text

# get text content
def get_text(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    all_text = ''
    for text in soup.body.find_all(string=True):
        if text.parent.name not in ['script', 'meta', 'style'] and not isinstance(text, Comment) and text.strip() != '':
            #print(text.strip(), '-----------------')
            all_text += ' '+text.strip()
    return all_text

def google_query(query):
    for res in search(query, num=5, stop=5):
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0",
                   "Accept": "text/html,application/xhtml+xml, application/xml;q=0.9,*/*;q=0.8",
                   "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding":"gzip, deflate", "DNT": "1",
                   "Connection": "close", "Upgrade-Insecure-Requests": "1"}
        r = requests.get(res, headers=headers)
        h = html2text.HTML2Text()
        h.ignore_links=True
        doc = h.handle(r.text)
        print(doc)
