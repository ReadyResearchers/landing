'''Test Module for File handling's methods'''

import resume_util.file_handling as fh

path = 'resume_util/data/sample_resume.pdf'

def test_check_path(path):
    assert fh.check_path(path) == True