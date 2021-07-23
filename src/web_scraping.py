# common installs
# pip install bs4

# common imports
from bs4 import BeautifulSoup, Comment
import requests

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