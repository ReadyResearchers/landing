import jd_handling as jd
import file_handling as fh
import text_mining as tm
import pandas as pd
import os

'''Load resume content'''
def get_resume_content(path):
    content = fh.load_data_pdf(path)
    return content

def main():
    df = jd.load()[0]
    im_df = jd.get_important_content(df)[1]
    im_dict = jd.get_important_content(df)[0]
    jd_content_list = jd.get_jd_content(im_dict)
    source_file = r'data/sample_resume.pdf'
    resume_path = os.path.join(os.getcwd(), source_file)
    resume_content = get_resume_content(resume_path)
    result_df = get_top_similarity(resume_content, jd_content_list, 3, im_df)
    print(result_df)

if __name__ == '__main__':
    main()