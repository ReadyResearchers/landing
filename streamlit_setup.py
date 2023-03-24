import lander.text_mining as tm
import lander.file_handling as fh

import streamlit as st
from streamlit_tags import st_tags
import base64
from PIL import Image
import time
import pandas as pd
import pandas as pd
import nltk
import PyPDF2
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
pd.options.mode.chained_assignment = None

st.set_page_config(
   page_title="Lander: A NLP-based Resume Analyzer",
   page_icon='image/lander.png',
)


@st.cache_data
def strip_cluster(data_eval, cluster, im_df):
    '''Function to slice the dataframe into given cluster'''
    ind = []
    for i in data_eval.index:
        if int(data_eval["ClusterName"][i]) == int(cluster):
            ind.append(i)
    match_df = im_df.loc[ind]
    return match_df

@st.cache_data
def match_score_cs(match_df, content):
    '''Caculate similarity score using cosine similarity, take top 100'''
    similarityscore = []
    for ind in match_df.index:
        score = tm.similarity_caculator(content, match_df['jobdescription'][ind])
        similarityscore.append(score)
    match_df['SimilarityScore'] = pd.Series(similarityscore)
    temp_df1 = match_df.copy()
    temp_df1 = temp_df1.sort_values('SimilarityScore', ascending=False)
    temp_df1 = temp_df1.iloc[:100]
    return temp_df1

@st.cache_data
def match_score_pm(data_eval, cluster, content, im_df, seniority):
    '''Caculate similarity score using Phrase Matcher'''
    match_df = strip_cluster(data_eval, cluster, im_df)
    if seniority.lower() in ['junior' or 'jr' or 'entry']:
        match_df = match_df[match_df["jobtitle"].str.contains("Senior|Sr|senior|Manager|Principal") == False]
    match_df = match_score_cs(match_df, content)
    scores = []
    matches_kws = []
    for ind in match_df.index:
        text_skill = match_df['Extracted Skills'][ind]
        text_skill_use = [k.lower() for k in text_skill.split(',')]
        matches_kw = tm.keyword_matching(content.lower(), text_skill_use)
        matches_kws.append(','.join(matches_kw))
        score = (len(matches_kw)/len(text_skill.split(',')))*100
        scores.append(score)

    match_df['MatchingPercentage'] = pd.Series(scores)
    match_df['KeywordMatched'] = pd.Series(matches_kws)
    return match_df

def main():
    img = Image.open('image/lander.png')
    st.image(img)
    st.sidebar.markdown("...Please Choose Something...")
    activities = ["Home", "Analyzer", "Database"]
    choice = st.sidebar.selectbox("Please select: Home to know more about the project" 
                                    + "\n Analyzer to analyze your resume" 
                                    + "\n Database to learn more about the dataset and download it if you need", activities)
    im_df = pd.read_csv('./data/skill_extracted_df.csv', index_col = 0)

    # prepare a data frame of only skills and job title to train
    col = ['jobtitle', 'Extracted Skills']
    data_eval = im_df[col]

    # Drop rows with missing data
    data_eval.dropna(subset=['Extracted Skills'], inplace=True)
    data_forfit = data_eval['Extracted Skills']

    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf = True, min_df = 0.001, use_idf=True, stop_words= 'english')

    tfidf_matrix = tfidf_vectorizer.fit_transform(data_forfit)

    # generate k-cluster
    num_clusters = 26
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.predict(tfidf_matrix)

    #add cluster name into the df
    data_eval["ClusterName"] = clusters

    if choice == 'Analyzer':

        # Load job description file
        
        
        # Collecting Miscellaneous Information
        act_name = st.text_input('Please enter your name')
        seniority = st.text_input('Please enter your seniority level in tech')


        # Upload Resume
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload Your Resume, And Get Smart Recommendations</h5>''',unsafe_allow_html=True)

        ## file upload in pdf format
        pdf_file = st.file_uploader("Please upload your Resume", type=["pdf"])
        if pdf_file is not None:
            ### parsing and extracting whole resume 
            content = fh.load_data_pdf(pdf_file)

            if content:

                ## Showing Analyzed data from (resume_data)
                st.header("Here is your Resume report")
                st.success("Hello "+ act_name)
                st.subheader("Below is your basic info")
                try:
                    st.text('Name: '+ act_name)                  
                    st.text('Seniority Level: '+ seniority)

                except:
                    pass
                

                st.subheader("Below is top 5 job matches your skills and info")
                cluster = km.predict(tfidf_vectorizer.transform([content])) 

                # Put all rows having matched cluster into a new dataframe with matching score
                match_df = match_score_pm(data_eval, cluster, content, im_df, seniority)
                
                # Return top 5:
                match_df = match_df.sort_values('MatchingPercentage', ascending=False)
                top_df = match_df.iloc[:5]
                
                # Return missing keyphrase from a job
                x = 1
                y = 10
                for ind in top_df.index:
                    st.success("🎆 Your skills is matched to " + top_df['jobtitle'][ind] + " 🎆")
                    key = top_df['Extracted Skills'][ind]
                    key_list = [k.lower() for k in key.split(',')]
                    matched_key = str(top_df['KeywordMatched'][ind])
                    matched_key_list = matched_key.split(',')
                    missing = []
                    for kw in key_list:
                        if kw not in matched_key_list:
                            missing.append(kw)
                    st_tags(label=' Your matched skills with this job are',
                    text='See our skills recommendation below',value=matched_key_list,key = x)
                    st_tags(label='### Recommended skills for you to boost chance with this job title',
                    text='Recommended skills generated from System',value= missing, key = y)
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h5>''',unsafe_allow_html=True)
                    x += 1
                    y += 1
                    with st.expander(label="Click to display Job Description"):
                        st.text(top_df['jobdescription'][ind])
                        
                    
    
    elif choice == 'Home':   

        st.subheader("Lander: Together we shoot for the moon")

        st.markdown('''
        <p align='justify'>
            A text-mining based tool to help student with finding the most compatible job post based on their past experiences and interesrs as well as optimizing their resume by a keyword suggesting system.
        </p>
        <p align="justify">
        <b>How to use it: -</b> <br/><br/>
        <b>Analyzer -</b> <br/>
            In the Side Bar select Analyzer option to start the process by filling out the required fields and uploading your resume in pdf format.<br/>
            Just sit back and relax our tool will do the magic on it's own.<br/><br/>
        <b>Feedback -</b> <br/>
            A place where user can suggest some feedback about the tool.<br/><br/>
        </p><br/><br/>
     
        ''',unsafe_allow_html=True)  
    elif choice == "Database":
        option = ['Download Database', "Cluster Plotting"]
        choice1 = st.selectbox(
            'What action do you want to choose', option)
        if choice1 == 'Download Database':
            st.subheader("You can download Lander's job and skills dataset below")
            with open("./data/skill_extracted_df.csv", "rb") as file:
                st.download_button(
                label="Download database",
                data=file,
                file_name="lander_skill_extracted_df.csv",
                mime="text/csv"
                )
            st.success("Now you can do something with this database!!!")
        elif choice1 == "Cluster Plotting":
            st.subheader("Preview top keyword in each clusters with relevant job titles")
            dfs = tm.get_top_features_cluster(tfidf_matrix.toarray(), clusters, 6, tfidf_vectorizer)
            for i in range(0,26):
                st.pyplot(tm.plotWords(dfs, 6, data_eval, i))
main()
