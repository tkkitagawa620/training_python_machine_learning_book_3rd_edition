from multiprocessing import Pipe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from base64 import encode
import encodings
from operator import index
import pandas as pd
# import pyprind
# import os
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# basepath = 'aclImdb'
# labels = {'pos': 1, 'neg': 0}
# pbar = pyprind.ProgBar(50000)
# df = pd.DataFrame()
# for s in ('test', 'train'):
#     for _l in ('pos', 'neg'):
#         path = os.path.join(basepath, s, _l)
#         for file in sorted(os.listdir(path)):
#             with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[_l]]], ignore_index=True)
#             pbar.update()

# df.columns = ['review', 'sentiment']
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('movie_data.csv', index=False, encoding='utf-8')
df = pd.read_csv('movie_data_preprocessed.csv', encoding='utf-8')


"""
レビューのテキスト情報からhtmlタグや絵文字(e.g ;) や :))を省く処理
"""


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)',
                           text)
    text = (re.sub('[\\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text
# df['review'] = df['review'].apply(preprocessor)
# df.to_csv('movie_data_preprocessed.csv', encoding='utf-8')


"""
トークン化
    -> 文章から単語一つ一つに区切る処理のことをいう。
    たとえば...
        "The sun is very bright"
        -> ["The", "sun", "is", "very", "bright"]

ステミング
    -> その際に現在形、過去形といったような形で使われている単語を原型に変形することで、関連する単語を同じ単語にマッピングできるようにする。
    一般的にはPorterステマーが使われるが、他にもSnowballステマーやLancasterステマーがある。
"""


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


"""
ストップワード
    -> is, the, hasといった単語は自然言語処理において有益な情報が含まれていることが少ない。それらの単語のことをストップワードという

"""
stop = stopwords.words('english')

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [stop, None],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "clf__penalty": ["11", "12"],
        "clf__C": [1.0, 10.0, 100.0]
    },
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [stop, None],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "vect__use_idf": [False],
        "vect__norm": [None],
        "clf__penalty": ["11", "12"],
        "clf__C": [1.0, 10.0, 100.0]
    }
]
lr_tfidf = Pipeline([("vect", tfidf), ("clf", LogisticRegression(random_state=0, solver='liblinear'))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)
