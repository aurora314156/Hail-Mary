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
        if self.model == 'SecondModel':
            guessAnswer = self.SecondModel()
        if self.model == 'SecondModelWithSoftmax':
            guessAnswer = self.SecondModelWithSoftmax()


        return guessAnswer
    
    def FirstModel(self):
        """
        merge story and question vector by add, calculate similarity with merge story and option vector
        """
        story = self.bc.encode([s_string])
        question = self.bc.encode([q_string])
        options = self.bc.encode(options)
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
    
    def SecondModel(self):
         """
        merge story and question vector by dot, calculate similarity with merge story and option vector
        """
        story = self.bc.encode([s_string])
        question = self.bc.encode([q_string])
        options = self.bc.encode(options)
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
    
    def SecondModelWithSoftmax(self):
         """
        merge story and question vector by dot, calculate similarity with merge story and option vector
        """
        story = self.bc.encode([s_string])
        question = self.bc.encode([q_string])
        options = self.bc.encode(options)
        merStoryQue = [x * y for x, y in zip(story, question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x * y for x, y in zip(story, option)]
            tmpScore = 1 - spatial.distance.cosine(self.softmax(merStoryQue), self.(merStoryOpt))
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def ThirdModel(self):
        """
        implementation original paper method
        """
        story = self.bc.encode([s_string])
        question = self.bc.encode([q_string])
        options = self.bc.encode(options)
        
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        merStoryQue = [x + y for x, y in zip(story, question)]
        for i in range(100):
            tmp = [x + y for x, y in zip(story, merStoryQue)]
            merStoryQue = tmp

        for option in self.options:
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, option)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)