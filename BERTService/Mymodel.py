from scipy import spatial
from TFIDF import TFIDF
from bert_serving.client import BertClient
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
        self.activationF = "softmax"
        self.constant = constant

    def MymodelMain(self):

        if self.model == 'FirstModel':
            guessAnswer = self.FirstModel(self.bc)
        if self.model == 'SecondModel':
            guessAnswer = self.SecondModel(self.bc)
        if self.model == 'ThirdModel':
            guessAnswer = self.ThirdModel(self.bc)
        if self.model == 'ForthModel':
            guessAnswer = self.ForthModel(self.bc)
        if self.model == 'FifthModel':
            guessAnswer = self.FifthModel(self.bc)
        if self.model == 'SixthModel':
            guessAnswer = self.SixthModel(self.bc)
        if self.model == 'SeventhModel':
            guessAnswer = self.SeventhModel(self.bc)
        if self.model == 'EighthModel':
            guessAnswer = self.EighthModel(self.bc)
        if self.model == 'NinthModel':
            guessAnswer = self.NinthModel(self.bc)
        if self.model == 'TenthModel':
            guessAnswer = self.TenthModel(self.bc)
        if self.model == 'EleventhModel':
            guessAnswer = self.EleventhModel(self.bc)
        if self.model == 'TwelfthModel':
            guessAnswer = self.TwelfthModel(self.bc)
        if self.model == 'ThirteenthModel':
            guessAnswer = self.ThirteenthModel(self.bc)
        if self.model == 'FourteenthModel':
            guessAnswer = self.FourteenthModel(self.bc)
        if self.model == 'FifteenthModel':
            guessAnswer = self.FifteenthModel(self.bc)
        if self.model == 'SixteenthModel':
            guessAnswer = self.SixteenthModel(self.bc)
        if self.model == 'SeventeenthModel':
            guessAnswer = self.SeventeenthModel(self.bc)
        if self.model == 'EighteenthModel':
            guessAnswer = self.EighteenthModel(self.bc)
        if self.model == 'NineteenthModel':
            guessAnswer = self.NineteenthModel(self.bc)
        if self.model == 'TwentiethModel':
            guessAnswer = self.TwentiethModel(self.bc)
        if self.model == 'TwentyFirstModel':
            guessAnswer = self.TwentiethModel(self.bc)
        if self.model == 'TestModel2':
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
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
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
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
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
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)
        

        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
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
            tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector)
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
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)


        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(story, option) + (options_tfscores[ind] * self.constant)
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
            tmpScore = 1 - spatial.distance.cosine(option, question)
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
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
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
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
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
            tmpScore = 1 - spatial.distance.cosine(story, option)
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
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
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
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)
        

        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
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
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)
        

        for option in options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
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
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)


        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option) + (options_tfscores[ind] * self.constant)
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
            tmpScore = 1 - spatial.distance.cosine(s, question)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore


        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        highestScore = 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector) + (options_tfscores[ind] * self.constant)
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
            tmpScore = 1 - spatial.distance.cosine(s, storyMerQue)
            if tmpScore > highestScore:
                highestScore_storyVector = s
                highestScore = tmpScore

        # test add tf-idf score
        options_tfscores = []

        for option in self.options:
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
                if o not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(o)]
            options_tfscores.append(tmp)

        highestScore = 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(option, highestScore_storyVector) + (options_tfscores[ind] * self.constant)
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
            tmp = 0
            for o in option.split(" "):
                o = o.lower()
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
            tmp = 0
            for word in sentence.split(" "):
                word = word.lower()
                if word not in self.TF_words:
                    continue
                tmp += self.TF_scores[0][self.TF_words.index(word)]
            sentences_tfscores.append(tmp)

        ind, guessAnswer, highestScore, highestScore_storyVector = 0, 0, 0, []

        for m in merQueOpts:
            s_ind = 0
            for s in storySentences:
                tmpScore = 1 - spatial.distance.cosine(s, m) + (sentences_tfscores[s_ind] * self.constant)
                if tmpScore > highestScore:
                    guessAnswer = ind
                    highestScore = tmpScore
                s_ind += 1
            ind += 1
        
        return guessAnswer
    
    def NineteenthModel(self, bc):
        
        merStoryQue = self.activationFunction(bc.encode([self.s_string + self.q_string]))
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
            for m in merQueOpts:
                tmpScore = 1 - spatial.distance.cosine(m, mSQO)
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
            for m in merQueOpts:
                tmpScore = 1 - spatial.distance.cosine(m, mSQ_QO)
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
                        sentences.append(sentence[:-1] + + self.q_string)
                    else:
                        sentences.append(sentence + + self.q_string)
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
        elif self.activationF == 'relu':
            return self.relu(x)
        elif self.activationF == 'drelu':
            return self.drelu(x)
        else:
            print("Activation function setting error.")
            return 0

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        
    def relu(self, x):
        x = np.asarray(x)
        x = x * (x > 0)
        return x

    def drelu(self, x):
        x = np.asarray(x)
        x = 1. * (x > 0)
        return x