from scipy import spatial

class Mymodel():
    def __init__(self, story, question, options):
        self.story = story
        self.question = question
        self.options = options
    
    def MymodelMain(self):
        
        merStoryQue = [x + y for x, y in zip(self.story, self.question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x + y for x, y in zip(self.story, self.option)]
            tmpScore = 1 - spatial.distance.cosine(dataSetI, dataSetII)
            print("tmpScore ",tmpScore)
            if tmpScore < highestScore:
                guessAnswer = ind
            ind += 1
        
        return guessAnswer