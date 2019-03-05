from scipy import spatial

class Mymodel():
    def __init__(self, story, question, options):
        self.story = story
        self.question = question
        self.options = options
    
    def MymodelMain(self):
        
        F_guessAnswer = self.FirstModel()
        
        return F_guessAnswer
        

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