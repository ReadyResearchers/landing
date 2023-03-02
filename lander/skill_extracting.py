'''This module extract the skill from job description and add those skill into the data frame for further clustering'''

# This module has been run on cloud in order to save memory space on the machine
# The python file is here to present what it does, it will not involve in the main process as it serve as data preprocessing procedure

import numpy as np
import pandas as pd
import re
import nltk
from rake_nltk import Rake
from spacy.matcher import PhraseMatcher, Matcher
import en_core_web_sm
import nltk
nltk.download('stopwords')

master = en_core_web_sm.load(disable = ['ner', 'tagger', 'parser'])

# Load job description file
df = pd.read_csv('dice_com_techjob_post.csv')
im_df = pd.DataFrame(df, columns = ['company', 'employment', 'jobdescription', 'jobtitle', 'skills'])
data_dict = im_df.to_dict()
jd_content = [x for x in data_dict['jobdescription'].values()]
skills = [x for x in data_dict['skills'].values()]

skills_cleaned = []

# Clean the skill data set by deleting meaningless skill and skill that doesn't make sense

for skill in skills:
    skill = str(skill)
    skill = skill.split(',')
    for entity in skill:
      if 'see' not in entity and 'See' not in entity and 'SEE' not in entity and re.search("^[a-zA-Z0-9.+-_][a-zA-Z0-9_ ]*[a-zA-Z0-9_]$", entity) and len(entity.split(" ")) < 5:
        skills_cleaned.append(entity)
        

#load skill set, convert to csv
df_skill = pd.read_excel('Technology Skills.xlsx')
extra_skill_df = pd.DataFrame(df_skill, columns = ['Example'])
extra_skill_dict = extra_skill_df.to_dict()
extra_skill_list = [x for x in extra_skill_dict['Example'].values()]

# combine two list of skills, take out duplicate

control = [x.lower() for x in skills_cleaned]
for skill in extra_skill_list:
  if skill.lower() not in control:
    skills_cleaned.append(skill)

'''Method to extract keyword'''
def keyword_matching(text_jd, text_skill):
    # Generate matcher pattern by extracting keywords from job description
    rake = Rake()
    matcher = PhraseMatcher(master.vocab)
    #rake.extract_keywords_from_text(text_jd)
    #jd_keyword = rake.get_ranked_phrases()
    patterns = [master.make_doc(k) for k in text_skill]
    matcher.add("Spec", patterns) 

    # Matching the keyword in job description with resume
    text_jd = master(text_jd)
    matches = matcher(text_jd)

    match_keywords = []
    for match_id, start, end in matches:
        kw = text_jd[start:end]
        if kw.text not in match_keywords:
          match_keywords.append(kw.text)

    return match_keywords

scores = []
matches_kws = []

for ind in im_df.index:
  text_jd = im_df['jobdescription'][ind]
  matches_kw = keyword_matching(text_jd, 
  skills_cleaned)
  matches_kws.append(','.join(matches_kw))

im_df['Extracted Skills'] = pd.Series(matches_kws)

# Export the extracted skill data frame into a csv file
extracted_df = pd.DataFrame(im_df, columns = ['company', 'employment', 'jobdescription', 'jobtitle', 'Extracted Skills'])
extracted_df.to_csv('skill_extracted_df.csv')