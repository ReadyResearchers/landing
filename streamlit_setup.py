import lander.text_mining as tm
import lander.file_handling as fh

import streamlit as st
from streamlit_tags import st_tags
import base64
from PIL import Image
from resume_parser import resumeparse
import time
import pandas as pd

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
        
        ## file upload in pdf format
        pdf_file = st.file_uploader("Please upload your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('...Please give us a second...'):
                time.sleep(4)
        
            ### saving the uploaded resume to folder
            uploaded_path = './Uploaded_resume/'+pdf_file.name
            pdf_name = pdf_file.name
            with open(uploaded_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(uploaded_path)

            ### parsing and extracting whole resume 
            resume_data = resumeparse.read_file('uploaded_path')
            if resume_data:
                
                ## Get the whole resume data into resume_text
                resume_text = fh.load_data_pdf(uploaded_path)

                ## Showing Analyzed data from (resume_data)
                st.header("Here is your Resume report")
                st.success("Hello "+ resume_data['name'])
                st.subheader("Below is your basic info")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['phone'])
                    st.text('Degree: '+str(resume_data['degree']))                    
                    st.text('Seniority Level: '+ seniority)

                except:
                    pass
                st.subheader("Below is top 5 job matches your skills and info")
                #TODO: Intergrate cluster
                
                keywords = st_tags(label=' Your current skills are',
                text='See our skills recommendation below',value=resume_data['skills'],key = '1  ')
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