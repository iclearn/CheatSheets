# common imports
import re
from nltk import sent_tokenize, word_tokenize

# remove non-ascii characters
def remove_non_ascii(text):
    text = text.encode('ascii',errors='ignore').decode()
    return text

# remove multiple spaces
def remove_excess_whitespace(text):
    text = re.sub('[ \t\n]{2, }', ' ' , text)
    return text