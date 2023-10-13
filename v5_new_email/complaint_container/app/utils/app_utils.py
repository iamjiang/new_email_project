import os
from pathlib import Path

def get_nltk_data_dir():
    project_dir = Path(__file__).absolute().parent.parent.parent
    nltk_data_dir = f'{project_dir}/resources/nltk/nltk_data'
    if not os.path.isdir(nltk_data_dir):
        nltk_data_dir = '/home/jpmcnobody/nltk_data'
    return nltk_data_dir
