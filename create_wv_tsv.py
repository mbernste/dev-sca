import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim.models import KeyedVectors as KV
import pandas as pd
import numpy as np
import json

VECS_DB = './GoogleNews-vectors-negative300.bin'

def main():

    # Load the model
    print('Loading model...')
    model = KV.load_word2vec_format(VECS_DB, binary=True)
    print('end')

    analog_df = pd.read_csv('BATS_3.0/3_Encyclopedic_semantics/E10 [male - female].txt', sep='\t', header=None)
    print(analog_df)
    
    m_to_f = {
        row[0]: row[1]
        for ri, row in analog_df.iterrows()
    }
    print(m_to_f)

    model_words = set(model.vocab.keys())

    words = []
    vecs = []
    for m, fs in m_to_f.items():
        fs = fs.split('/')
        for f in fs:
            if m in model_words and f in model_words:
                words.append(m)
                words.append(f)
                vecs.append(model[m])
                vecs.append(model[f])
    
    df = pd.DataFrame(
        data=vecs,
        index=words
    )
    df.to_csv('gendered_word_vectors.tsv', sep='\t')

if __name__ == '__main__':
    main()
