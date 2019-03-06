from scipy import spatial
from bert_serving.client import BertClient
import numpy as np

class Mymodel():
    def __init__(self, bc, s_string, q_string, options):
        self.bc = bc
        self.story = bc.encode([story])
        self.question = bc.encode([q_string])
        self.options = bc.encode(options)
    
    def MymodelMain(self):
        
        #F_guessAnswer = self.FirstModel()
        S_guessAnswer = self.SecondModel()

        return S_guessAnswer
        
    def FirstModel(self):
        merStoryQue = [x + y for x, y in zip(self.story, self.question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in self.options:
            merStoryOpt = [x + y for x, y in zip(self.story, option)]
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer
    
    def SecondModel(self):
        tmp, ind, guessAnswer, highestScore = [], 0, 0, 0
        merStoryQue = [x + y for x, y in zip(self.story, self.question)]
        for i in range(19):
            tmp = [x + y for x, y in zip(self.story, merStoryQue)]
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