import file_handling as fh
import pandas as pd
import os 

source_file = r'data/dice_com_techjob_post.csv'
global path 
path = os.path.join(os.getcwd(), source_file)

'''Load the job data file and return data frame and dictionary of data'''
def load():
    df = fh.load_data_csv(path)
    df_dict = fh.load_data_dictionary(df)
    return df, df_dict

'''Load important content in the input file and store in a dictionary'''
def get_important_content(df):
    im_df = pd.DataFrame(df, columns = ['company', 'employment', 'jobdescription', 'jobtitle', 'skills'])
    im_dict = fh.load_data_dictionary(im_df)
    return im_dict, im_df

'''Load all job description into a list'''
def get_jd_content(im_dict):
    jd_content = [x for x in im_dict['jobdescription'].values()]
    return jd_content