
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class TFIDF():
    def __init__(self, s_string, q_string, options):
        self.s_string = s_string
        self.q_string = q_string
        self.options = options

    def getTFIDFWeigths(self):
        corpus = []
        corpus.append(self.s_string)
        corpus.append(self.q_string)
        for o in self.options:
            corpus.append(o)

        #将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        #计算个词语出现的次数
        X = vectorizer.fit_transform(corpus)
        #获取词袋中所有文本关键词
        word = vectorizer.get_feature_names()
        print(word)
        #查看词频结果
        print(X.toarray())
        #类调用
        transformer = TfidfTransformer()
        #print(transformer)
        #将词频矩阵X统计成TF-IDF值
        tfidf = transformer.fit_transform(X)
        # get words and scores
        tfidf_word = vectorizer.get_feature_names()
        tfidf_scores = tfidf.toarray()
        
        for i in range(len(weight)):
            print(u"-------这里输出第",i,u"类文本的词语tf-idf权重------")
            for j in range(len(word)):
                print("word: {}, score: {}".format(word[j],weight[i][j]))

        print("done")

        return tfidf_word, tfidf_scores