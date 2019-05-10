from scipy import spatial
from scipy.spatial.distance import euclidean, hamming, cityblock, minkowski
from TFIDF import TFIDF
from bert_serving.client import BertClient
import time
from scipy.special import softmax
import numpy as np


class Mymodel():
    def __init__(self, bc, s_string, q_string, options, m, TF_words, TF_scores, constant):
        self.bc = bc
        self.model = m
        self.s_string = s_string
        self.q_string = q_string
        self.options = options
        self.stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        self.TF_words = TF_words
        self.TF_scores = TF_scores
        self.activationF = "softmax3"
        self.similarity = euclidean
        self.constant = constant

    def MymodelMain(self):

        if self.model == 'FirstModel':
            guessAnswer = self.FirstModel(self.bc)
        elif self.model == 'SecondModel':
            guessAnswer = self.SecondModel(self.bc)
        elif self.model == 'ThirdModel':
            guessAnswer = self.ThirdModel(self.bc)
        elif self.model == 'ForthModel':
            guessAnswer = self.ForthModel(self.bc)
        elif self.model == 'FifthModel':
            guessAnswer = self.FifthModel(self.bc)
        elif self.model == 'SixthModel':
            guessAnswer = self.SixthModel(self.bc)
        elif self.model == 'SeventhModel':
            guessAnswer = self.SeventhModel(self.bc)
        elif self.model == 'EighthModel':
            guessAnswer = self.EighthModel(self.bc)
        elif self.model == 'NinthModel':
            guessAnswer = self.NinthModel(self.bc)
        elif self.model == 'TenthModel':
            guessAnswer = self.TenthModel(self.bc)
        elif self.model == 'EleventhModel':
            guessAnswer = self.EleventhModel(self.bc)
        elif self.model == 'TwelfthModel':
            guessAnswer = self.TwelfthModel(self.bc)
        elif self.model == 'ThirteenthModel':
            guessAnswer = self.ThirteenthModel(self.bc)
        elif self.model == 'FourteenthModel':
            guessAnswer = self.FourteenthModel(self.bc)
        elif self.model == 'FifteenthModel':
            guessAnswer = self.FifteenthModel(self.bc)
        elif self.model == 'SixteenthModel':
            guessAnswer = self.SixteenthModel(self.bc)
        elif self.model == 'SeventeenthModel':
            guessAnswer = self.SeventeenthModel(self.bc)
        elif self.model == 'EighteenthModel':
            guessAnswer = self.EighteenthModel(self.bc)
        elif self.model == 'NineteenthModel':
            guessAnswer = self.NineteenthModel(self.bc)
        elif self.model == 'TwentiethModel':
            guessAnswer = self.TwentiethModel(self.bc)
        elif self.model == 'TwentyFirstModel':
            guessAnswer = self.TwentyFirstModel(self.bc)
        elif self.model == 'TwentySecondModel':
            guessAnswer = self.TwentySecondModel(self.bc)
        elif self.model == 'TwentyThirdModel':
            guessAnswer = self.TwentyThirdModel(self.bc)
        elif self.model == 'TwentyForthModel':
            guessAnswer = self.TwentyForthModel(self.bc)
        elif self.model == 'TwentyFifthModel':
            guessAnswer = self.TwentyFifthModel(self.bc)
        elif self.model == 'TwentySixthModel':
            guessAnswer = self.TwentySixthModel(self.bc)
        elif self.model == 'TwentySeventhModel':
            guessAnswer = self.TwentySeventhModel(self.bc)
        elif self.model == 'TwentyEighthModel':
            guessAnswer = self.TwentyEighthModel(self.bc)
        elif self.model == 'TestModel2':
            guessAnswer = self.TestModel(self.bc)
        
        return guessAnswer
    
    def FirstModel(self,bc):
        """
        merge story and question vector by add, calculate similarity with merge story and option vector
        """
        story = self.activationFunction(bc.encode([self.s_string]))
        question = self.activationFunction(bc.encode([self.q_string]))
        options = self.activationFunction(bc.encode(self.options))
        merStoryQue = [x + y for x, y in zip(story, question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x + y for x, y in zip(story, option)]
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
            tmpScore = angle_between(merStoryQue, merStoryOpt)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SecondModel(self, bc):
        """
        merge story and question vector by dot, calculate similarity with merge story and option vector
        """
        story = self.activationFunction(bc.encode([self.s_string]))
        question = self.activationFunction(bc.encode([self.q_string]))
        options = self.activationFunction(bc.encode(self.options))
        merStoryQue = [x * y for x, y in zip(story, question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x * y for x, y in zip(story, option)]
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
            tmpScore = angle_between(merStoryQue, merStoryOpt)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def ThirdModel(self, bc):
        """
        implementation original paper method
        """
        story = self.activationFunction(bc.encode([self.s_string]))
        question = self.activationFunction(bc.encode([self.q_string]))
        options = self.activationFunction(bc.encode(self.options))
        
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        merStoryQue = [x + y for x, y in zip(story, question)]
        for i in range(20):
            tmp = [x + y for x, y in zip(story, merStoryQue)]
            merStoryQue = tmp

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)
        
        for option in merQueOpts:
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            tmpScore = angle_between(merStoryQue, option)
            #tmpScore =  self.similarity(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer


    def ForthModel(self, bc):
        """
        encode story sentences, then use each story sentences vector to calculate similarity with question
        choose highest score story vector to calculate similarity with options
        """
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1])
                    else:
                        sentences.append(sentence)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)

        story_sentences = self.activationFunction(bc.encode(sentences))
        question = self.activationFunction(bc.encode([self.q_string]))
        options = self.activationFunction(bc.encode(self.options))
        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        for s in story_sentences:
            tmpScore = 1 - spatial.distance.cosine(s, question)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore

        highestScore = 0
        for option in options:
            #tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector)
            tmpScore = angle_between(highestScore_storyVector, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1

        return guessAnswer

    def FifthModel(self, bc):
        
        merStoryQue = self.activationFunction(bc.encode([self.q_string + self.s_string]))
        options = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SixthModel(self, bc):
        story = self.activationFunction(bc.encode([self.s_string]))
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string 
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag ==3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)


        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merQueOpts:
            tmpScore = angle_between(story, option)
            #tmpScore = 1 - spatial.distance.cosine(story, option) + (options_tfscores[ind] * self.constant)
            #tmpScore = self.similarity(story, option) + (options_tfscores[ind] * self.constant)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SeventhModel(self, bc):
        question = self.activationFunction(bc.encode([self.q_string]))
        # merge story and options
        for o in range(len(self.options)):
            self.options[o] = self.options[o] + self.s_string
        merStoryOpt = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merStoryOpt:
            #tmpScore = 1 - spatial.distance.cosine(option, question)
            tmpScore = angle_between(question, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def EighthModel(self, bc):
        
        story = self.activationFunction(bc.encode([self.s_string]))
        question = self.activationFunction(bc.encode([self.q_string]))
        merStoryQue = [x + y for x, y in zip(story, question)]
        options = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            tmpScore = angle_between(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def NinthModel(self, bc):
        merStoryQue = self.activationFunction(bc.encode([self.q_string + self.s_string]))
        # merge story and options
        for o in range(len(self.options)):
            self.options[o] = self.options[o] + self.s_string
        merStoryOpt = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merStoryOpt:
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            tmpScore = angle_between(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def TenthModel(self, bc):
        story = self.activationFunction(bc.encode([self.s_string])) 
        options = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            #tmpScore = 1 - spatial.distance.cosine(story, option)
            tmpScore = angle_between(story, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def EleventhModel(self, bc):
        merStoryQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merQueOpts:
            tmpScore = angle_between(merStoryQue, option)
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            #tmpScore = self.similarity(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def TwelfthModel(self, bc):

        story = self.activationFunction(bc.encode([self.s_string]))
        question = self.activationFunction(bc.encode([self.q_string]))
        
        merStoryQue = [x + y for x, y in zip(story, question)]
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        for i in range(200):
            tmp = [x + y for x, y in zip(merStoryQue, question)]
            merStoryQue = tmp

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)
        
        for option in merQueOpts:
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            tmpScore = angle_between(merStoryQue, option)
            #tmpScore = self.similarity(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def ThirteenthModel(self, bc):

        story = self.activationFunction(bc.encode([self.s_string]))
        question = self.activationFunction(bc.encode([self.q_string]))
        
        merStoryQue = [x + y for x, y in zip(story, question)]
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        for i in range(200):
            tmp = [x + y for x, y in zip(merStoryQue, question)]
            merStoryQue = tmp

        options = self.activationFunction(bc.encode(self.options))

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)
        
        for option in options:
            tmpScore = angle_between(merStoryQue, option)
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            #tmpScore = self.similarity(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def FourteenthModel(self, bc):

        merStoryQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))
        question = self.activationFunction(bc.encode([self.q_string]))

        merStoryQue = [x + y for x, y in zip(merStoryQue, question)]
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        for i in range(200):
            tmp = [x + y for x, y in zip(merStoryQue, question)]
            merStoryQue = tmp

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        for option in merQueOpts:
            tmpScore = angle_between(merStoryQue, option)
            #tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            #tmpScore = self.similarity(merStoryQue, option) + (options_tfscores[ind] * self.constant)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1

        return guessAnswer

    def FifteenthModel(self, bc):
        """
        encode eacht story sentences with question sentence, then calculate similarity with wholte story and question
        choose highest vector to calculate similarity with question and options
        """

        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1] + self.q_string)
                    else:
                        sentences.append(sentence + self.q_string)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)

        storySentencesMerQuestion = self.activationFunction(bc.encode(sentences))
        question = self.activationFunction(bc.encode([self.q_string]))

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        for s in storySentencesMerQuestion:
            #tmpScore = 1 - spatial.distance.cosine(s, question)
            tmpScore = angle_between(s, question)
            #tmpScore = self.similarity(s, question)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        highestScore = 0
        for option in merQueOpts:
            tmpScore = angle_between(highestScore_storyVector, option)
            #tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector) + (options_tfscores[ind] * self.constant)
            #tmpScore = self.similarity(option, highestScore_storyVector) + (options_tfscores[ind] * self.constant)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1

        return guessAnswer

    def SixteenthModel(self, bc):
        """
        encode eacht story sentences with question sentence, then calculate similarity with wholte story and question
        choose highest vector to calculate similarity with question and options
        """
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1] + self.q_string)
                    else:
                        sentences.append(sentence + self.q_string)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)

        storySentencesMerQuestion = self.activationFunction(bc.encode(sentences))
        storyMerQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        for s in storySentencesMerQuestion:
            #tmpScore = 1 - spatial.distance.cosine(s, storyMerQue)
            tmpScore = angle_between(storyMerQue, s)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        highestScore = 0
        for option in merQueOpts:
            #tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector) + (options_tfscores[ind] * self.constant)
            tmpScore = angle_between(highestScore_storyVector, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1

        return guessAnswer
    
    def SeventeenthModel(self, bc):
        """
        encode eacht story sentences with question sentence, then calculate similarity with wholte story and question
        choose highest vector to calculate similarity with question and options
        """
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1])
                    else:
                        sentences.append(sentence)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)

        storySentences = self.activationFunction(bc.encode(sentences))
        question = self.activationFunction(bc.encode([self.q_string]))

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        for s in storySentences:
            tmpScore = 1 - spatial.distance.cosine(s, question)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        highestScore = 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1

        return guessAnswer

    def EighteenthModel(self, bc):
        """
        encode eacht story sentences, then calculate similarity with question and options
        choose highest scores options as final answer
        """
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1])
                    else:
                        sentences.append(sentence)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)
        
        storySentences = self.activationFunction(bc.encode(sentences))
        
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string +self.s_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

         # test add tf-idf score
        sentences_tfscores = []

        for sentence in sentences:
            tmp, flag = 0, 0
            for word in sentence.split(" "):
                word = word.lower()
                if flag == 3:
                    break
                if word not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(word)]
            sentences_tfscores.append(tmp)

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        for m in merQueOpts:
            s_ind = 0
            for s in storySentences:
                tmpScore = 1 - spatial.distance.cosine(s, m) + (sentences_tfscores[s_ind] * self.constant)
                #tmpScore = self.similarity(s, m) + (sentences_tfscores[s_ind] * self.constant)
                if tmpScore > highestScore:
                    guessAnswer = ind
                    highestScore = tmpScore
                s_ind += 1
            ind += 1
        
        return guessAnswer
    
    def NineteenthModel(self, bc):
        
        #merStoryQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))
        merStoryQue = self.activationFunction(bc.encode([self.s_string]))
        options = self.activationFunction(bc.encode(self.options))

        merStoryQueOpt, ind, guessAnswer, highestScore, o_ind = [], 0, 0, 0, 0
        
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        for m in merQueOpts:
            tmpStoryQueOpt = merStoryQue
            for i in range(50):
                tmp = [x + y for x, y in zip(tmpStoryQueOpt, m)]
                tmpStoryQueOpt = tmp
            merStoryQueOpt.append(tmpStoryQueOpt)

        for mSQO in merStoryQueOpt:
            ind = 0
            for m in merQueOpts:
                tmpScore = 1 - spatial.distance.cosine(m, mSQO)
                #tmpScore = self.similarity(m, mSQO)
                if tmpScore > highestScore:
                    guessAnswer = ind
                    highestScore = tmpScore
                ind += 1
        
        return guessAnswer
    
    def TwentiethModel(self, bc):
        
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))
        merStoryQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))
        
        merStoryQue_QueOpt, ind, guessAnswer, highestScore, o_ind = [], 0, 0, 0, 0

        for m in merQueOpts:
            tmpStoryQue = merStoryQue
            for i in range(50):
                tmp = [x + y for x, y in zip(tmpStoryQue, m)]
                tmpStoryQue = tmp
            merStoryQue_QueOpt.append(tmpStoryQue)
        
        merStoryQue_QueOpt = self.activationFunction(merStoryQue_QueOpt)

        for mSQ_QO in merStoryQue_QueOpt:
            ind = 0
            for m in merQueOpts:
                tmpScore = 1 - spatial.distance.cosine(m, mSQ_QO)
                #tmpScore = self.similarity(m, mSQ_QO)
                if tmpScore > highestScore:
                    guessAnswer = ind
                    highestScore = tmpScore
                ind += 1
        
        return guessAnswer

    def TwentyFirstModel(self,bc):

        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1])
                    else:
                        sentences.append(sentence)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)
        
        storySentences = self.activationFunction(bc.encode(sentences))

        merStoryQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        for s in storySentences:
            tmpScore = 1 - spatial.distance.cosine(s, merStoryQue)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore

        # # test add tf-idf score
        # options_tfscores = []

        # for option in self.options:
        #     tmp = 0
        #     for o in option.split(" "):
        #         o = o.lower()
        #         if o not in self.TF_words:
        #             continue
        #         tmp += self.TF_scores[0][self.TF_words.index(o)]
        #     options_tfscores.append(tmp)

        highestScore = 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector) 
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1

        return guessAnswer

    def TwentySecondModel(self,bc):
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1] + self.q_string)
                    else:
                        sentences.append(sentence + self.q_string)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)
        
        storySentencesMerQuestion = self.activationFunction(bc.encode(sentences))

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0

        for m in merQueOpts:
            for sMQ in storySentencesMerQuestion:
                tmpScore = 1 - spatial.distance.cosine(m, sMQ)
                #tmpScore = self.similarity(m, sMQ)
                if tmpScore > highestScore:
                    guessAnswer = ind
                    highestScore = tmpScore
            ind += 1
        
        return guessAnswer
        
    def TwentyThirdModel(self,bc):
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        for o in self.options:
                            sentences.append(sentence[:-1] + self.q_string + o)
                    else:
                        for o in self.options:
                            sentences.append(sentence + self.q_string  + o)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)
        
        storySentencesMerQuestionAndOpts = self.activationFunction(bc.encode(sentences))
        
        Que = self.activationFunction(bc.encode([self.q_string]))

        ind, guessAnswer, highestScore = 0, 0, 0

        for s in storySentencesMerQuestionAndOpts:
            tmpScore = 1 - spatial.distance.cosine(Que, s)
            #tmpScore = self.similarity(Que, s)
            if tmpScore > highestScore:
                guessAnswer = ind % 4
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def TwentyForthModel(self,bc):
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        for o in self.options:
                            sentences.append(sentence[:-1] + o)
                    else:
                        for o in self.options:
                            sentences.append(sentence + o)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)
        
        storySentencesMerOpts = self.activationFunction(bc.encode(sentences))
        
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))
        #Que = self.activationFunction(bc.encode([self.q_string]))
        
        ind, guessAnswer, highestScore = 0, 0, 0

        for m in merQueOpts:
            for sMO in storySentencesMerOpts:
                tmpScore = 1 - spatial.distance.cosine(m, sMO)
                #tmpScore = self.similarity(m, sMO)
                if tmpScore > highestScore:
                    guessAnswer = ind % 4
                    highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def TwentyFifthModel(self,bc):
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1])
                    else:
                        sentences.append(sentence)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)

        storySentences = self.activationFunction(bc.encode(sentences))
        question = self.activationFunction(bc.encode([self.q_string]))
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        storySentencesDict = {}
        for s in storySentences:
            storySentencesDict[1 - spatial.distance.cosine(s, question)] = s
        
        sortedSentenceDict = [(k,storySentencesDict[k]) for k in sorted(storySentencesDict.keys(), reverse=True)]
        
        guessAnswer, highestScore = 0, 0

        for sv in sortedSentenceDict[:10]:
            ind = 0
            for o in merQueOpts:
                tmpScore = 1 - spatial.distance.cosine(sv[1], o)
                if tmpScore >= highestScore:
                    guessAnswer = ind
                    highestScore = tmpScore
                ind += 1

        return guessAnswer
    
    def TwentySixthModel(self,bc):
        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1])
                    else:
                        sentences.append(sentence)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)
        
        storySentences = self.activationFunction(bc.encode(sentences))
        question = self.activationFunction(bc.encode([self.q_string]))
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        ind, guessAnswer, highestScore = 0, 0, 0

        for m in merQueOpts:
            tmpScore = 0
            tmpScorelist = []
            for s in storySentences:
                tmpScorelist.append(1 - spatial.distance.cosine(m, s))
                tmpScorelist.sort(reverse=True)
            for t in tmpScorelist[:5]:
                tmpScore += t
            if tmpScore > highestScore:
                highestScore = tmpScore
                guessAnswer = ind
            ind += 1

        return guessAnswer

    def TwentySeventhModel(self,bc):
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        storyAttQue = self.AttOverAtt(story, question)
        opts = bc.encode(self.options)

        ind, guessAnswer, highestScore = 0, 0, 0

        for o in opts:
            tmpScore = 1 - spatial.distance.cosine(storyAttQue, o)
            if tmpScore > highestScore:
                highestScore = tmpScore
                guessAnswer = ind
            ind += 1
            
        return guessAnswer

    def TwentyEighthModel(self,bc):
        """
        encode eacht story sentences with question sentence, then calculate similarity with wholte story and question
        choose highest vector to calculate similarity with question and options
        """

        sentences, tmp_string, sentence = [], "", ""
        for s in self.s_string[:len(self.s_string)-1]:
            tmp_string += s
            # reserve sentence structure
            if s == "." or s == "?" or s == "!":
                # remove "," "." "?"
                sentence = ""
                for t in tmp_string:
                    if t is "," or t is "." or t is "?":
                        continue
                    else:
                        sentence += t
                if len(sentence) >1:
                    if sentence[0] == " ":
                        sentences.append(sentence[:-1] + self.q_string)
                    else:
                        sentences.append(sentence + self.q_string)
                tmp_string = ""
                continue

        # use whole story structure
        if tmp_string != "":
            sentences.append(tmp_string)

        storySentencesMerQuestion = self.activationFunction(bc.encode(sentences))
        question = self.activationFunction(bc.encode([self.q_string]))

        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        storyAndQueAtt = []
        for s in storySentencesMerQuestion:
            storyAndQueAtt.append(self.AttOverAtt(s, question))

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp, flag = 0, 0
            for o in option.split(" "):
                o = o.lower()
                if flag == 3:
                    break
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        highestScore = 0

        for option in merQueOpts:
            for att in storyAndQueAtt:
                tmpScore = 1 - spatial.distance.cosine(option, att) + (options_tfscores[ind] * self.constant)
                #tmpScore = self.similarity(option, highestScore_storyVector) + (options_tfscores[ind] * self.constant)
                if tmpScore > highestScore:
                    guessAnswer = ind
                    highestScore = tmpScore
            ind += 1

        return guessAnswer


    def TestModel(self,bc):

        merStoryQue = self.s_string + self.q_string
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + merStoryQue
        
        merStoryQueOpts = self.activationFunction(bc.encode(self.options))

        merStoryQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))

        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        for option in merStoryQueOpts:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def TestModel2(self,bc):

        question = self.activationFunction(bc.encode([self.q_string]))
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string 
        
        merQueOpts = self.activationFunction(bc.encode(self.options))

        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(option, question)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def activationFunction(self, x):
        if self.activationF == 'softmax':
            return self.softmax(x)
        elif self.activationF == 'softmax2':
            return self.softmax2(x)
        elif self.activationF == 'softmax3':
            return self.softmax3(x)
        elif self.activationF == 'relu':
            return self.relu(x)
        elif self.activationF == 'drelu':
            return self.drelu(x)
        else:
            print("Activation function setting error.")
            return 0

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        def soft(xxx):
            return (np.exp(xxx) / np.sum(np.exp(xxx)))
        tmp = []
        for xx in x:
            tmp.append(soft(xx))

        return tmp
    
    def softmax2(self,x):
        """improvement version"""
        if len(x) == 1:
            exps = np.exp(x - np.max(x))
            res = (exps / np.sum(exps))
            return res
        else:
            tmp = []
            for xx in x:
                exps = np.exp(xx - np.max(xx))
                res = (exps / np.sum(exps))
                tmp.append(res)
            return tmp
    
    def softmax3(self, x):
        """Compute softmax values for each sets of scores in x."""
        x = softmax(x)
        return x
    
    def relu(self, x):
        tmp = []
        for xx in x:
            xx = xx * (xx > 0)
            tmp.append(xx)
        return tmp

    def drelu(self, x):
        x = np.asarray(x)
        x = 1. * (x > 0)
        return x

    def AttOverAtt(self, doc, query):

        # Individual ATT layer
        if isinstance(doc, list):
            doc = np.asarray([doc])
        if isinstance(query, list):
            query = np.asarray(query)
        if len(doc) != 1:
            doc = np.reshape(doc, (1,len(doc)))
        
        matrix = np.matmul(doc.T, query)
        # row-wise softmax matrix
        rowWiseSoftmax, columnWiseSoftmax = [], []
        for c in matrix:
            rowWiseSoftmax.append(self.activationFunction(c))
        # column-wise softmax matrix
        for r in matrix.transpose():
            columnWiseSoftmax.append(self.activationFunction(r))
        # column-wise average matrix
        columnWiseAveMatrix, attentionOverAttention = [], []
        # for c in range(len(rowWiseSoftmax)):
        #     tmp = 0
        #     for r in range(len(rowWiseSoftmax)):
        #         tmp+=rowWiseSoftmax[c][r]
        #     columnWiseAveMatrix.append(np.average(tmp))
        rowWiseSoftmax_transpose = list(map(list, zip(*rowWiseSoftmax)))
        for r in rowWiseSoftmax_transpose:
            columnWiseAveMatrix.append(np.average(r))
        # final dot product
        # for r in range(len(columnWiseSoftmax)):
        #     tmp = 0
        #     for c in range(len(columnWiseSoftmax)):
        #         tmp += columnWiseSoftmax[r][c] * columnWiseAveMatrix[c]
        #     attentionOverAttention.append(tmp)
        for c in columnWiseSoftmax:
            attentionOverAttention.append(np.dot(c, columnWiseAveMatrix))
        
        return attentionOverAttention


    # angle similarity
    def unit_vector(self, vector):
    """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))