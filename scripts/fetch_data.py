import logging

import wget

logging.basicConfig(level=logging.INFO)

DATA_URL = 'https://github.com/temmyzeus/No-Toxic/blob/master/data/jigsaw-toxic-comment-classification-challenge.zip?raw=true'
DATA_DIR = '../data_test'

logging.info(f'Fetching data from {DATA_URL}')
logging.info('Downloading File...')
filename = wget.download(DATA_URL, out=DATA_DIR)
logging.info(f'File Downloaded to {filename}')
