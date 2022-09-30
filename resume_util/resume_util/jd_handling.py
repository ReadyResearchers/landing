import file_handling as fh
import pandas as pd

path = 'data/dice_job_techjob_post.csv'

'''Load the job data file and return data frame and dictionary of data'''
def load(path):
    if fh.check_path(path):
        df = fh.load_data_csv(path)
        df_dict = fh.load_data_dictionary(df)
    return df, df_dict

'''Load important content in the input file and store in a dictionary'''
def get_important_content():
    df = load(path)[0]
    im_df = pd.DataFrame(df, columns = ['company', 'employment', 'jobdescription', 'jobtitle', 'skills'])
    im_dict = fh.load_data_dictionary(im_df)
    return im_dict

'''Load all job description into a list'''
def get_jd_content():
    im_dict = get_important_content()
    jd_content = [x for x in im_dict['jobdescription']]
    return jd_content