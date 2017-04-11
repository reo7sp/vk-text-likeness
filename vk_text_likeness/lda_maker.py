import gensim
import nltk
from gensim.models.ldamulticore import LdaMulticore
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords


nltk.download('stopwords')
ru_stopwords = set(stopwords.words('russian'))

with open('data/stopwords.txt') as f:
    for line in f:
        ru_stopwords.add(line.lower().strip())


class LdaMaker:
    def __init__(self, corpora, num_topics):
        self.num_topics = num_topics

        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.stemmer = RussianStemmer()

        corpora_tokenzied = [self.tokenizer.tokenize((self._keep_only_russian_chars(str(doc).lower()))) for doc in corpora]

        corpora_stemmed = []
        for doc in corpora_tokenzied:
            stemmed_doc = [self.stemmer.stem(token) for token in doc if token not in ru_stopwords]
            stemmed_doc = [token for token in stemmed_doc if token not in ru_stopwords]
            corpora_stemmed.append(stemmed_doc)

        self.dictionary = gensim.corpora.Dictionary(corpora_stemmed)
        corpora_bow = [self.dictionary.doc2bow(doc) for doc in corpora_stemmed]
        # self.tfidf = gensim.models.TfidfModel(corpora_bow)
        # corpora_tfidf = self.tfidf[corpora_bow]
        self.lda = LdaMulticore(num_topics=self.num_topics, corpus=corpora_bow, id2word=self.dictionary)

    def get(self, doc):
        doc = self.tokenizer.tokenize(self._keep_only_russian_chars(doc.lower()))
        doc = [self.stemmer.stem(token) for token in doc if token not in ru_stopwords]
        doc = [token for token in doc if token not in ru_stopwords]
        doc = self.dictionary.doc2bow(doc)
        # doc = self.tfidf[doc]
        return self.lda[doc]

    @staticmethod
    def _keep_only_russian_chars(s):
        new_s = ''
        for c in s:
            if 'а' <= c <= 'я' or 'А' <= c <= 'Я':
                new_s += c
            else:
                new_s += ' '
        return new_s
