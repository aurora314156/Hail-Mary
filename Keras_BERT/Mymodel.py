from scipy import spatial
import numpy as np

class Mymodel():
    def __init__(self, model, token_dict, s_string, q_string, options, model_id):
        self.model = model
        self.token_dict = token_dict
        self.model_id = model_id
        self.s_string = s_string
        self.q_string = q_string
        self.options = options
    
    def MymodelMain(self):
        guessAnswer = ""
        #guessAnswer = self.FirstModel()
        if self.model_id == 'ForthModel':
            print(self.model_id)
            guessAnswer = self.ForthModel()
        if self.model_id == 'FirstModel':
            guessAnswer = self.FirstModel()
        if self.model_id == 'SecondModel':
            guessAnswer = self.SecondModel()
        
        return guessAnswer
    
    def FirstModel(self):
        """
        merge story and question vector by add, calculate similarity with merge story and option vector
        """
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
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
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
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
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
        merStoryQue = [x * y for x, y in zip(story, question)]
        ind, guessAnswer, highestScore = 0, 0, 0
        for option in options:
            merStoryOpt = [x * y for x, y in zip(story, option)]
            tmpScore = 1 - spatial.distance.cosine(self.softmax(merStoryQue), self.softmax(merStoryOpt))
            if tmpScore > highestScore:
                guessAnswer = ind
                highestScore = tmpScore
            ind += 1
        
        return guessAnswer

    def ThirdModel(self):
        """
        implementation original paper method
        """
        story = bc.encode([self.s_string])
        question = bc.encode([self.q_string])
        options = bc.encode(self.options)
        
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
    
    def ForthModel(self):
        """
        encode story sentences, then use each story sentences vector to calculate similarity with question
        choose highest score story vector to calculate similarity with options
        """
        print("run forthModel")
        sentences = self.getParsetResStrToList()
        sentences, question, options = self.preprocess(sentences, self.q_string, self.options)

        story_sentences, options = [], []
        for s in sentences:
            story_setences.append(self.getSentenceEmbedWithPool(s))
        for o in options:
            options.append(self.getSentenceEmbedWithPool(o))

        print(story_sentences.shape)
        print(options.shape)

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
    
    
    def getSentenceEmbedWithPool(self, tokens):
        print("get sentence embed with pooling")
        token_input = np.asarray([[self.token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
        seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])

        print('Inputs:', token_input[0][:len(tokens)])
        predicts = model.predict([token_input, seg_input])[0]
        print('Pooled:', predicts.tolist()[:5])
        
        return predicts

    def getParserResStrToList(self):
        print("Start parser string to list")
        sentences, tmp_string = [], ""
        for s in self.s_string:
            if s == "." or s == "?":
                sentences.append(tmp_string)
                tmp_string = ""
                continue
            tmp_string += s

        # use whole story structure
        if tmp_string != "":
            sentences.append(sentences[:len(tmp_string)-1])
        
        return sentences
    
    def preprocess(self, sentences, question, options):
        """
        due to original code limit
        """
        print("start preprocess data vector")
        question = ['[CLS]'] + question
        question.append('[SEP]')
        tmp_sentences = []
        for sentence in sentences:
            sentence = ['[CLS]'] + sentence
            sentence.append('[SEP]')
            tmp_sentences.append(sentence)
        tmp_options = []
        for option in options:
            option = ['[CLS]'] + option
            option.append('[SEP]')
            tmp_options.append(option)

        return tmp_sentences, question, tmp_options

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)