import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import nltk
from nltk.stem import PorterStemmer
import PyPDF
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import jd_handling as jd

# Load job description file
df = pd.read_csv('dice_com_techjob_post.csv')
im_df = pd.DataFrame(df, columns = ['company', 'employment', 'jobdescription', 'jobtitle', 'skills'])

jd_content = [x for x in data_dict['jobdescription'].values()]

skills_cleaned = []

df = jd.load()[0]
im_df = jd.get_important_content(df)[1]
data_dict = im_df.to_dict()

def clean_skills_list(data_dict):
  # Create a list of skills from database
  skills = [x for x in data_dict['skills'].values()]
  for skill in skills:
      skill = str(skill)
      skill = skill.split(',')
      for entity in skill:
        if 'see' not in entity and 'See' not in entity and 'SEE' not in entity:
          skills_cleaned.append(entity)
  return skills_cleaned

# Eliminate all row contain see description in skills by extracting skills from job description:
def skills_overwrite(im_df, skills_cleaned):
  for i in im_df.index:
    skill_raw = str(im_df['skills'][i])
    skill_list = []
    if len((skill_raw.split(' '))) < 4:
      for word in im_df['jobdescription'][i]:
        if word in skills_cleaned:
          skill_list.append(word)
    im_df['skills'][i] = ' '.join(skill_list)


def df_prepare(im_df):
  # prepare a data frame of only skills and job title to train
  col = ['jobtitle', 'skills']
  data_eval = im_df[col]
  # Drop rows with missing data
  data_eval.dropna(subset=['skills'], inplace=True)

  for i in data_eval.index:
    skill = str(data_eval['skills'][i])
    skill = skill.replace(',', '')
    data_eval['skills'][i] = skill

  data_forfit = data_eval['skills']
  return data_forfit

def clustering(data_forfit, data_eval, num_clusters):
# define vectorizer parameters
  tfidf_vectorizer = TfidfVectorizer(sublinear_tf = True, min_df = 0.001, use_idf=True, stop_words= 'english')
  tfidf_matrix = tfidf_vectorizer.fit_transform(data_forfit)
  # generate k-cluster
  km = KMeans(n_clusters=num_clusters)
  km.fit(tfidf_matrix)
  clusters = km.predict(tfidf_matrix)
  #add cluster name into the df
  data_eval["ClusterName"] = clusters
  return data_eval, clusters

def prediction(tfidf_vectorizer, resume_content, km, data_eval, im_df):
  cluster = km.predict(tfidf_vectorizer.transform([resume_content])) #you have to use transform only and not fit_transfrom 
  ind = []
  for i in data_eval.index:
    if int(data_eval["ClusterName"][i]) == int(cluster):
      ind.append(i)
  match_df = im_df.loc(ind)
  return match_df


def get_top_features_cluster(tf_idf_array, prediction, n_feats, tfidf_vectorizer):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = tfidf_vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def get_job(data_eval):
  # Get job at a certain clusters
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
    for i in range(0, len(dfs)):
        plt.figure(figsize=(8, 2))
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.figtext(0.99, - 0.2, f'Associated positions: {job[i][:3]}', horizontalalignment='right')
        

#labels = cluster_name(clusters, data_eval)
#print(labels, " ")
dfs = get_top_features_cluster(tfidf_matrix.toarray(), clusters, 6)
plotWords(dfs, 6, data_eval)

!pip install gensim

# Switch to LDA approach
from gensim import corpora, models, similarities

!pip install rake_nltk

from rake_nltk import Rake

# Get keyword only from jd_content using Rake
rake = Rake()
def get_kw_rake(jd_content):
  for jd in jd_content:
    rake.extract_keywords_from_text(jd)
  keywords = rake.get_ranked_phrases()
  return keywords
