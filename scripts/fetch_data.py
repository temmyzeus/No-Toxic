import os
import logging

import wget

logging.basicConfig(level=logging.INFO)

DATA_URL = 'https://github.com/temmyzeus/No-Toxic/blob/master/data/jigsaw-toxic-comment-classification-challenge.zip?raw=true'
DATA_DIR = './data'

logging.info(f'Fetching data from {DATA_URL}')
logging.info('Downloading File...')

if not os.path.exists(DATA_DIR):
    logging.info('{} not found, downloading to current directory'.format(DATA_DIR))
    filename = wget.download(DATA_URL)
else:
    filename = wget.download(DATA_URL, out=DATA_DIR)

logging.info(f'File Downloaded to {filename}')