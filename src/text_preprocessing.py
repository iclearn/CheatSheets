# common imports
import re
from nltk import sent_tokenize, word_tokenize


# common text cleaning commands
def preprocess(text):
    # remove non-ascii characters
    text = text.encode('ascii',errors='ignore').decode()
    # remove multiple white-spaces
    text = re.sub('[ \t\n]{2, }', ' ' , text)
    # remove phrases introduced by pdfminer
    text = re.sub(r'\(cid:[0-9]+\)', ' ', text)
    text = re.sub(u'\xa0', ' ', text) # space
    text = re.sub('(\so\s)|(\s\uf0b7\s)', ' | ', text) # bullet points

    return text

def get_paragraphs(text):
    # splits text on presence of multiple \n
    # remove step from pre-processing to convert multiple whitespace to single whitespace
    return re.split(r'[\n][ \r\t]*[\n]+', text)


def is_header(text):
    ans = False
    caps_count = 0
    total_count = 0
    w_list = word_tokenize(text)
    if w_list[-1] == '.':
        # it's a sentence
        return ans
    # remove special characters
    w_list = [w for w in w_list if w.isalnum()]
    if len(w_list) < 8 and len(w_list) != 0:
        for w in w_list:
            if w.isupper() or w.istitle() or w.isnumeric():
                caps_count += 1
            total_count += 1
        if float(caps_count)/total_count > 0.5:
            # it is a header
            ans = True
    return ans