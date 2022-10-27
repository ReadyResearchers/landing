'''Home for text mining methods'''
import string
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher, Matcher
from spacy.util import filter_spans
from collections import Counter
from rake_nltk import Rake
import en_core_web_sm
import pandas
 
# Load spaCy trained pipeline for english
master = en_core_web_sm.load()

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
def keyword_matching(text_resume, text_jd):
    # Generate matcher pattern by extracting keywords from job description
    rake = Rake()
    matcher = PhraseMatcher(master.vocab)
    jd_keyword = rake.extract_keywords_from_text(text_resume)
    jd_keyword_count = Counter(jd_keyword)
    patterns = [master.make_doc(k) for k in jd_keyword]
    matcher.add("Spec", patterns) 

    # Matching the keyword in job description with resume
    text_resume = master(text_resume)
    matches = matcher(text_resume)
    match_keywords = [text_resume[start:end] for _, start, end in matches]

    # Count the amount of word matched and matching frequency
    matcher_report = Counter(match_keywords)

    # Calculate the keyword matching percentage between job description and resume
    matched_amount = len(matcher_report.keys())
    jd_keyword_amount = len(jd_keyword_count.keys())
    matcher_percentage = (matched_amount/jd_keyword_amount)*100

    return matcher_report, matcher_percentage

'''Calculate the similarity and return a list of a certain amount of job with highest score'''
def get_top_similarity(resume_content, jd_content_list, im_df):
    max = 0
    max_id = 0
    rank = 1
    top_dict = {}
    id_list = []
    # Get the rank and score of job description compare to resume
    while(amount > 0):
        for jd_content in jd_content_list:
            similarity_score = tm.similarity_caculator(resume_content, jd_content)
            if similarity_score > max:
                max = similarity_score
                max_id = jd_content_list.index(jd_content)
        top_dict['Rank'] = rank
        top_dict['Score']= max
        id_list.append[max_id]
        jd_content_list.pop[max_id]
        amount -= 1
        rank += 1
    
    top_df = pd.DataFrame(top_dict)
    im_df = im_df.reset_index()
    im_df_rs = pd.DataFrame()

    for index, row in im_df.iterrows():
        for id in id_list:
            if id == index:
                im_df_rs = im_df_rs.append(row, ignore_index = True)

    frames = [top_df,im_df_rs]
    result_df = pd.concat(frames)
    return top_dict