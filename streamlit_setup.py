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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


nltk.download('punkt')
st.set_page_config(
   page_title="Lander: A NLP-based Resume Analyzer",
   page_icon='image/lander.png',
)

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    # (Logo, Heading, Sidebar etc)
    img = Image.open('image/lander.png')
    st.image(img)
    st.sidebar.markdown("...Please Choose Something...")
    activities = ["Home", "Analyzer", "Feedback"]
    choice = st.sidebar.selectbox("Please select Home to know more about the project, Analyzer to analyze your resume and Feedback to give me some feedback:", activities)

    if choice == 'Analyzer':
        # Collecting Miscellaneous Information
        act_name = st.text_input('Please enter your name')
        act_mail = st.text_input('Please enter your email')
        seniority = st.text_input('Please enter your seniority level in tech')


        # Upload Resume
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload Your Resume, And Get Smart Recommendations</h5>''',unsafe_allow_html=True)
        

        # Load job description file
        df = pd.read_csv('./data/dice_com_techjob_post.csv', index_col = 0)
        im_df = pd.read_csv('./data/skill_extracted_df.csv', index_col = 0)
        data_dict = im_df.to_dict()
        skills = [x for x in data_dict['Extracted Skills'].values()]

        # prepare a data frame of only skills and job title to train
        col = ['jobtitle', 'Extracted Skills']
        data_eval = im_df[col]

        # Drop rows with missing data
        data_eval.dropna(subset=['Extracted Skills'], inplace=True)
        data_forfit = data_eval['Extracted Skills']
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf = True, min_df = 0.01, use_idf=True, stop_words= 'english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(data_forfit)

        # define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf = True, min_df = 0.001, use_idf=True, stop_words= 'english')

        tfidf_matrix = tfidf_vectorizer.fit_transform(data_forfit)

        # generate k-cluster

        num_clusters = 26

        km = KMeans(n_clusters=num_clusters)

        km.fit(tfidf_matrix)

        clusters = km.predict(tfidf_matrix)

        ## file upload in pdf format
        pdf_file = st.file_uploader("Please upload your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('...Please give us a second...'):
                time.sleep(2)

            ### parsing and extracting whole resume 
            pdfReader = PyPDF2.PdfReader(pdf_file)
            page = pdfReader.pages[0]
            content = page.extract_text()

            if content:

                ## Showing Analyzed data from (resume_data)
                st.header("Here is your Resume report")
                st.success("Hello "+ act_name)
                st.subheader("Below is your basic info")
                try:
                    st.text('Name: '+ act_name)
                    st.text('Email: ' + act_mail)                    
                    st.text('Seniority Level: '+ seniority)

                except:
                    pass
                st.subheader("Below is top 5 job matches your skills and info")

                #add cluster name into the df
                data_eval["ClusterName"] = clusters
                cluster = km.predict(tfidf_vectorizer.transform([content])) 
                ind = []
                for i in data_eval.index:
                    if int(data_eval["ClusterName"][i]) == int(cluster):
                        ind.append(i)
                match_df = im_df.loc[ind]

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
                # Return top 5:
                temp_df = match_df.copy()
                top_df = match_df.copy()
                amount = 5
                max_ind = []
                while(amount > 0):
                    max = 0
                    max_id = 0
                    for ind in temp_df.index:
                        if temp_df['MatchingPercentage'][ind] > max:
                            max_id = ind
                            max = temp_df['MatchingPercentage'][ind]

                    max_ind.append(max_id)
                    temp_df['MatchingPercentage'][max_id] = 0
                    amount -= 1
                
                # Return missing keyphrase from a job
                for ind in max_ind:
                    print('-------------------- Missing Key Phrase Report -------------------------------')
                    print("Report of missing key phrase for job: ", top_df['jobtitle'][ind])
                    key = top_df['Extracted Skills'][ind]
                    key_list = [k.lower() for k in key.split(',')]
                    matched_key = top_df['KeywordMatched'][ind]
                    matched_key_list = matched_key.split(',')
                    print('List of every key skill in the job:', key)
                    missing = []
                    for kw in key_list:
                        if kw not in matched_key_list:
                            missing.append(kw)
                    print()
                    print(f'Matched key phrase: {matched_key}')
                    print()
                    i = 1
                    for ms in missing:
                        print(f'Top {i} missing phrase: {ms}')
                        print()
                        i += 1
                        if i > 40:
                            break
                    print('---------------------------End of report--------------------------------------')
                    print()
                
                #keywords = st_tags(label=' Your current skills are',
                #text='See our skills recommendation below',value=resume_data['skills'],key = '1  ')
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
main()