import os
import pandas as pd
import PyPDF2

'''Check if the input path is valid'''
def check_path(path):
    if os.path.exists(path):
        return True
    else:
        return False

'''Load into data frame for visualization'''
def load_data_csv(path):
    df = pd.read_csv(path)
    return df

'''Load into dictionary for text handling'''
def load_data_dictionary(df):
    data_dict = df.to_dict()
    return data_dict

'''Extract text from PDF file'''
def load_data_pdf(path):
    fileOb = open(path,'rb')
    pdfReader = PyPDF2.PdfFileReader(fileOb)
    page = pdfReader.getPage(0)
    content = page.extractText()
    return content

def load_data_other(path):
    with open(path, 'r') as f:
        content = f.read()
        return content   