from scipy import spatial
import numpy as np

class Mymodel():
    def __init__(self, story, question, options):
        self.story = story
        self.question = question
        self.options = options
    
    def MymodelMain(self):
        
        F_guessAnswer = self.FirstModel()
        
        return F_guessAnswer
        
    def FirstModel(self):
        merStoryQue = [x + y for x, y in zip(self.softmax(self.story), self.softmax(self.question))]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in self.options:
            merStoryOpt = [x + y for x, y in zip(self.softmax(self.story), self.softmax(option))]
            tmpScore = 1 - spatial.distance.cosine(merStoryQue, merStoryOpt)
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)