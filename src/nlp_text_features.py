# common imports
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import spacy
import textacy
from spacy import displacy
import gpt_2_simple as gpt2

# other imports
import pickle
import pandas as pd
import numpy as np

# Reusable constants/pre-trained models
stopwords.words('english')
nlp = spacy.load('en_core_web_md')

def generate_answers(context, question):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='default')
    # use checkpoint for fine-tuned gpt model
    ans = gpt2.generate(sess, run_name='default', checkpoint_dir='ckpt',#, model_name='355M', model_dir='models'
        length=200,
        prefix=context+' '+question,
        truncate='Q:', # to not generate new questions
        return_as_list=True,
        nsamples=1,
        temperature=0.1,
        seed=500,
        top_k=1)
    return ans[0]

def sentence_transformer_emb(text, text_list):
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    sen_emb = model.encode(text, convert_to_tensor=True)
    list_of_other_sen_emb = model.encode(text_list, convert_to_tensor=True)
    # similarity syntax
    sim_scores = util.pytorch_cos_sim(sen_emb, list_of_other_sen_emb)[0].cpu()
    top_res = np.argpartition(-sim_scores, range(1))[0] # to get index of top result


def extract_entities_spacy(text):
    ent_doc = nlp(text)
    ent = [(e.start_char, e.end_char, e.label_) for e in ent_doc.ents]
    return ent

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

