import jd_handling as jd
import file_handling as fh
import text_mining as tm
import pandas as pd

'''Load resume content'''
def get_resume_content(path):
    content = fh.load_data_pdf(path)
    return content

'''Calculate the similarity and return a list of a certain amount of job with highest score'''
def get_top_similarity(resume_content, jd_content_list, amount, im_df):
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
    return result_df

def main():
    im_df = jd.get_important_content()[1]
    jd_content_list = jd.get_jd_content()
    resume_path = 'data/Mai Nguyen_September 2022_SWE ver-2.pdf'
    resume_content = get_resume_content(resume_path)
    result_df = get_top_similarity(resume_content, jd_content_list, 3, im_df)
    print(result_df)

if '__name__' == '__main__':
    main()