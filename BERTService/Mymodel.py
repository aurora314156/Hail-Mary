from scipy import spatial
from bert_serving.client import BertClient
import numpy as np

class Mymodel():
    def __init__(self, bc, s_string, q_string, options, m):
        self.bc = bc
        self.model = m
        self.s_string = s_string
        self.q_string = q_string
        self.options = options
       
    def MymodelMain(self):
        #guessAnswer = self.FirstModel()
        if self.model == 'FirstModel':
            guessAnswer = self.FirstModel(self.bc)
        if self.model == 'SecondModel':
            guessAnswer = self.SecondModel(self.bc)
        if self.model == 'ThirdModel':
            guessAnswer = self.ThirdModel(self.bc)
        if self.model == 'ThirdModelWithSoftmax':
            guessAnswer = self.ThirdModelWithSoftmax(self.bc)
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

        return guessAnswer
    
    def FirstModel(self,bc):
        """
        merge story and question vector by add, calculate similarity with merge story and option vector
        """
        story = self.softmax(bc.encode([self.s_string]))
        question = self.softmax(bc.encode([self.q_string]))
        options = self.softmax(bc.encode(self.options))
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
        story = self.softmax(bc.encode([self.s_string]))
        question = self.softmax(bc.encode([self.q_string]))
        options = self.softmax(bc.encode(self.options))
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
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
        
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        merStoryQue = [x + y for x, y in zip(story, question)]
        for i in range(20):
            tmp = [x + y for x, y in zip(story, merStoryQue)]
            merStoryQue = tmp

        for option in options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def ThirdModelWithSoftmax(self, bc):
        """
        implementation original paper method
        """
        story = self.softmax(bc.encode([self.s_string]))
        question = self.softmax(bc.encode([self.q_string]))
        options = self.softmax(bc.encode(self.options))
        
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        merStoryQue = [x + y for x, y in zip(story, question)]
        for i in range(20):
            tmp = [x + y for x, y in zip(story, merStoryQue)]
            merStoryQue = self.softmax(np.asarray(tmp))

        for option in options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
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

        story_sentences = self.softmax(bc.encode(sentences))
        question = self.softmax(bc.encode([self.q_string]))
        options = self.softmax(bc.encode(self.options))
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
        
        merStoryQue = self.softmax(bc.encode([self.s_string + self.q_string]))
        options = self.softmax(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SixthModel(self, bc):
        story = self.softmax(bc.encode([self.s_string]))
        for i in range(len(self.options)):
            self.options[i] = self.q_string + self.options[i]
        
        merQueOpts = self.softmax(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(story, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SeventhModel(self, bc):
        question = self.softmax(bc.encode([self.q_string]))
        # merge story and options
        for o in range(len(self.options)):
            self.options[o] = self.options[o] + self.s_string
        merStoryOpt = self.softmax(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merStoryOpt:
            tmpScore = 1 - spatial.distance.cosine(option, question)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def EighthModel(self, bc):
        
        story = self.softmax(bc.encode([self.s_string]))
        question = self.softmax(bc.encode([self.q_string]))
        merStoryQue = [x + y for x, y in zip(story, question)]
        options = self.softmax(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def NinthModel(self, bc):
        merStoryQue = self.softmax(bc.encode([self.s_string + self.q_string]))
        # merge story and options
        for o in range(len(self.options)):
            self.options[o] = self.options[o] + self.s_string
        merStoryOpt = self.softmax(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merStoryOpt:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def TenthModel(self, bc):
        story = self.softmax(bc.encode([self.s_string])) 
        options = self.softmax(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            tmpScore = 1 - spatial.distance.cosine(story, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def EleventhModel(self, bc):
        merStoryQue = self.softmax(bc.encode([self.s_string + self.q_string]))
        for i in range(len(self.options)):
            self.options[i] = self.options[i] + self.q_string
        
        merQueOpts = self.softmax(bc.encode(self.options))

        ind, guessAnswer, highestScore = 0, 0, 0
        for option in merQueOpts:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        
    def relu(self, x):
        return x * (x > 0)

    def drelu(self, x):
        return 1. * (x > 0)