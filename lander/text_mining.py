'''Home for text mining methods'''
import string
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher, Matcher
from spacy.util import filter_spans
from collections import Counter
import en_core_web_sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Load spaCy trained pipeline for english
master = en_core_web_sm.load(disable=["tagger", "parser","ner"])

'''Return lemmatized text for better handling later'''
def lemmatization(text):
    text = master(text)
    token_list = []
    for token in text:
        token_list.append(token.lemma_)

    lemmatized_text = ' '.join(token_list)
    return lemmatized_text

'''Return a breakdown lemmatized token'''
def tokenizer(text):
    lemmatized_text = lemmatization(text)

    # remove punctuation from the text
    pre_processed = "".join(c for c in lemmatized_text if c not in string.punctuation)
    token_list =[]

    # Strip off unwanted punctuation
    for token in pre_processed:
        if token.is_stop is False:
            token_list.append(token)

    return token_list

'''Computer the frequency of certain keywords/words in a list'''
def freq_counter(list):
    word_freq = Counter(list)
    return word_freq

'''Extract all verb phrase for strength evaluation later on'''
def pos_verb(text):
    text = master(text)

    # Possible pattern for verb
    pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'AUX', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]

    # Create a Matcher instance
    matcher = Matcher(master.vocab)
    matcher.add("verb-phrase", None, pattern)
    # Matching verb phrase, return matchID with index
    matches = matcher(text)
    
    # Turn matches to text verb phrase and filter out duplicates/overlap
    pos_v = [text[start:end] for _, start, end in matches]
    filter_spans(pos_v)
    return pos_v

'''Compute the similarity between job description and resume using Cosine Similarity'''
def similarity_caculator(text_resume, text_jd):
    text_resume = lemmatization(text_resume)
    text_jd = lemmatization(text_jd)
    text_list = [text_resume, text_jd]
    cv = CountVectorizer()
    count_vector = cv.fit_transform(text_list)
    matchPercentage = cosine_similarity(count_vector)[0][1]*100
    matchPercentage = round(matchPercentage,2)
    return matchPercentage

"Compute the phrase matching for raw text in resume and jd"
def keyword_matching(text_resume, skill):
    # Generate matcher pattern by extracting keywords from job description
    matcher = PhraseMatcher(master.vocab)
    patterns = [master(k) for k in skill]
    matcher.add("Skill pattern", patterns) 

    # Matching the keyword in job description with resume
    text_resume = master(text_resume)
    matches = matcher(text_resume)
    match_keywords = []

    for match_id, start, end in matches:
      kw = text_resume[start:end]
      if kw.text not in match_keywords:
        match_keywords.append(kw.text)
   

    return match_keywords


def cluster_name(prediction, data_eval):
  labels = []
  for cluster in prediction:
    title_list = []
    for i in data_eval.index:
      if int(data_eval['ClusterName'][i]) == cluster:
        for word in data_eval['jobtitle'][i].split(' '):
          title_list.append(word)
    occurence_count = Counter(title_list)
    label = occurence_count.most_common(1)[0][0]
    labels.append(label)
  return labels

def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = tfidf_vectorizer.get_feature_names_out()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def get_job(data_eval):

  dict_job = {}

  for ind in data_eval.index:
    i = data_eval['ClusterName'][ind]
    if i not in dict_job.keys():
      dict_job[i] = []
      dict_job[i].append(data_eval['jobtitle'][ind])
    else:
      dict_job[i].append(data_eval['jobtitle'][ind])
  
  return dict_job


def plotWords(dfs, n_feats, data_eval):
    job = get_job(data_eval)
    for i in range(0,26):
        plt.figure(figsize=(8, 2))
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.figtext(0.99, - 0.2, f'Associated positions: {job[i][:3]}', horizontalalignment='right')