import jd_handling as jd
import file_handling as fh
from landing.resume_util.resume_util.file_handling import load_data_other
import text_mining as tm
import pandas as pd

def get_resume_content(path):
    content = load_data_pdf(path)