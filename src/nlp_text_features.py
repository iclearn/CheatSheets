# common imports
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import spacy
import textacy

stopwords.words('english')
nlp = spacy.load('en_core_web_md')

# other imports
import pickle
import pandas as pd

def get_transformer_emb(model_name, text_list, save_loc):
    st_model = SentenceTransformer(model_name)

    sentence_embeddings = st_model.encode(text_list)
    with open(save_loc, 'wb') as f:
        pickle.dump(sentence_embeddings, f)


def get_textacy_stats(stats, text_list):
    # stats = ['automated_readability_index', 'coleman_liau_index', 'flesch_kincaid_grade_level', 'flesch_reading_ease',
    #          'gunning_fog_index', 'lix', 'smog_index']
    docs = nlp.pipe(text_list)
    text_stats = {}
    ts = [textacy.text_stats.TextStats(doc) for doc in docs]
    for s in stats:
        text_stats[s] = [getattr(t, s) for t in ts]
    return text_stats

def get_tf_idf(text_list):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_list)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return df

