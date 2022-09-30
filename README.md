# Landing

A text-mining based tool to help student with finding the most compatible job post based on their past experiences and interesrs as well as optimizing their resume by a keyword suggesting system.

## Introduction & Motivation

With the detrimental effect of Covid-19, the job market has become extremely competitive for the last 3 years, especially for newly graduated candidates. During the summer of my freshman year, I was working as an intern for Talent Acquisition Service Office of a renowned bank back in Vietnam. In there, I have witnessed so many fresh candidates at a renowned bank in Vietnam. In there, I have witnessed so many fresh candidates who apply to a myriad of jobs without actually knowing what they want to do or what they are good at. Then, when I came back to America, a lot of seniors back then also tell me that they had the same problem. At that time, I realize that the struggle in job hunting is a worldwide problem.

Thus, my dream since then was to create a platform where fresh candidates can find what is the most suitable job they should apply to based on their past experiences and interests. In addition, I want to help them tailor their resume based on the job description of their dream job to enhance their chance of getting through the initial scanning round.  

## Technical Details and Future Plan

Python will be the main language that I use to implement this artifact, and I will deliver the result via streamlit or netlify depends on which one has a better interface for this tools. 

In order to match job description with resume, I will use keywords extraction feature from gensim, phrase.matcher from spacy, and cosine-similarity from sklearn. Before that, I will try my best to group word/skills that are coherently matched/equal/refer to each other by training a model on job description using gpt-2 or tensorflow.

In order to recommend which key word the student should incorporate to their resume, I extract the keywords in the job description using gensim and I will rank the importance of the keyword by an algorithm that I will design later on

## References


