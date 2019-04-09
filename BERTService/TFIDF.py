
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from ContentParser import ContentParser
import string

class TFIDF():
    def __init__(self, dataset):
        self.dataset = dataset
        self.stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    def getTFIDFWeigths(self):
        
        corpus = self.getListOfAllData()
        #将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        #计算个词语出现的次数
        X = vectorizer.fit_transform(corpus)
        #获取词袋中所有文本关键词
        word = vectorizer.get_feature_names()
        #print(word)
        #查看词频结果
        #print(X.toarray())
        #类调用
        transformer = TfidfTransformer()
        #print(transformer)
        #将词频矩阵X统计成TF-IDF值
        tfidf = transformer.fit_transform(X)
        # get words and scores
        tfidf_word = vectorizer.get_feature_names()
        tfidf_scores = tfidf.toarray()

        # normalizer
        norm1 = Normalizer(norm='l2')
        normalizer_scores = norm1.fit_transform(tfidf_scores)
        
        # for i in range(len(tfidf_scores)):
        #     print(u"-------这里输出第",i,u"类文本的词语tf-idf权重------")
        #     for j in range(len(tfidf_word)):
        #         if tfidf_scores[i][j] > 0:
        #             print("word: {}, score: {}".format(tfidf_word[j],tfidf_scores[i][j]))

        # print("done")

        return tfidf_word, normalizer_scores

    def getListOfAllData(self):

        corpus = []
        for single_dataset in self.dataset:
            if isinstance(single_dataset, str):
                continue
            for single_data in single_dataset:
                s_string, q_string, options, answer = ContentParser(single_data).getContent()
                tmp = ""
                tmp = s_string + q_string
                for o in options:
                    tmp += o
                print(tmp)
                # remove stop words
                tmp = self.remove_punctuation(tmp)
                
                oneCorpusContent = ""
                for t in tmp.lower().split(" "):
                    if t in self.stop_words:
                        continue
                    else:
                        oneCorpusContent += oneCorpusContent + " "
                print("one CorpusContent len: ", len(oneCorpusContent))
                corpus.append(oneCorpusContent[:-1])
        return corpus

    def remove_punctuation(s):
        table = string.maketrans("","")
        return s.translate(table, string.punctuation)