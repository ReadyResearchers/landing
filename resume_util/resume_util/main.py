import jd_handling as jd
import file_handling as fh
from landing.resume_util.resume_util.file_handling import load_data_other
import text_mining as tm
import pandas as pd

'''Load resume content'''
def get_resume_content(path):
    content = fh.load_data_pdf(path)
    return content

'''Calculate the similarity and return a list of a certain amount of job with highest score'''
def get_top_similarity(resume_content, jd_content_list, amount, im_dict):
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
    
    for key, values in im_dict.items()

    






