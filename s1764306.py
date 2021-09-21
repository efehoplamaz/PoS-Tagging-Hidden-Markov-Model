import inspect, sys, hashlib

# Hack around a warning message deep inside scikit learn, loaded by nltk :-(
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk

from nltk.corpus import brown
from nltk.tag import map_tag, tagset_mapping
import math
import numpy as np

if map_tag('brown', 'universal', 'NR-TL') != 'NOUN':
    # Out-of-date tagset, we add a few that we need
    tm=tagset_mapping('en-brown','universal')
    tm['NR-TL']=tm['NR-TL-HL']='NOUN'

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.01 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.

    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        #raise NotImplementedError('HMM.emission_model')
        # TODO prepare data
        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences

        data = [(tag, word.lower()) for sentence in train_data for (word, tag) in sentence]

        # TODO compute the emission mod
        emission_FD = nltk.ConditionalFreqDist(data)
        estimator = lambda freqDist : nltk.LidstoneProbDist(freqDist, 0.01, freqDist.B()+1)
        self.emission_PD = nltk.ConditionalProbDist(emission_FD, estimator)# nltk.LidstoneProbDist, 0.01, emission_FD.N())
        for tag, word in data:
            if tag not in self.states:
                self.states.append(tag)
        return self.emission_PD, self.states

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self,state,word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        #raise NotImplementedError('HMM.elprob')
        return self.emission_PD[state].logprob(word) # fixme

    # Compute transition model using ConditionalProbDist with a LidstonelprobDist estimator.
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        #raise NotImplementedError('HMM.transition_model')
        # TODO: prepare the data
        data = []

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        for s in train_data:
            s = [('<s>','<s>')] + s + [('</s>','</s>')]
            for i, (word, tag) in enumerate(s):
                if i == len(s)-1:
                    break
                data.append((s[i][1], s[i+1][1]))
        # TODO compute the transition model

        transition_FD = nltk.ConditionalFreqDist(data)
        estimator = lambda freqDist : nltk.LidstoneProbDist(freqDist, 0.01, freqDist.B()+1)
        self.transition_PD = nltk.ConditionalProbDist(transition_FD, estimator)

        return self.transition_PD

    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self,state1,state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        #raise NotImplementedError('HMM.tlprob')
        return self.transition_PD[state1].logprob(state2) # fixme

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """
        #raise NotImplementedError('HMM.initialise')

        # selected this data type because it will have s many columns
        # which will be dynamically allocating the elements

        # For viterbi data structure, I stored the costs in a List of Lists. This data strucutre
        # has s many lists, where s is the number of states, and each list holds t many elements
        # where t is the number of observations i.e number of words. (s by t two dimensional array)
        self.viterbi = []

        # For backpointer data structure, I stored the indexes which has the least cost in transition
        # from that particular state to another in a List of Lists. Again this data strucutre
        # has s many lists, where s is the number of states, and each list holds t many elements
        # where t is the number of observations i.e number of words. (s by t two dimensional array)
        self.backpointer = []


        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (-log-base-2 probabilities)
        # TODO

        for state in self.states:
            self.viterbi.append([-self.elprob(state, observation)-self.tlprob('<s>', state)])
        # Initialise step 0 of backpointer
        # TODO
            self.backpointer.append([0])
    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        #For each word in observations, the
        for t in range(1,len(observations)): # fixme to iterate over steps
            for s in range(len(self.states)): # fixme to iterate over states
                #  Use costs, not probabilities
                # For each Viterbi path probabilities in a step behind, calculate the maximum (in our case
                # minimum because of negative log probabilities i.e costs) probability of being in that state.
                # Probability of being in state is calculated by subtracting logprob of emissionPD and
                # logprob of transitionPD from the previous path probabilities.
                # self.states[s] will indicate which state we would like to go and self.viterbi[i][t - 1] will
                # loop thorugh all Viterbi path probabilities in a step behind.

                # We create a list of costs, which will store all costs from all states in a step behind to the
                # state self.states[s].
                total_prob = [(self.viterbi[i][t - 1] - self.elprob(self.states[s], observations[t]) - self.tlprob(self.states[i], self.states[s])) for i, state in enumerate(self.states)]

                # We will get the minimum of the list so that, we will minimize the cost while traveling to the next
                # state. Viterbi will hold the costs.
                self.viterbi[s].append(min(total_prob))

                # We will get the index of the minimum argument of the list so that,
                # we will minimize the cost while traveling to the next state.
                # Backpointer will hold the indexes.
                self.backpointer[s].append(np.argmin(total_prob))
        # TODO
        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        # We will not store the transitions from last word to </s>, instead we will have the index of state
        # which minimizes the cost from last word to </s>. To do this again I created a list of costs. This time
        # I did not add emission probability because all of the emission probability for states are constant.
        # (self.elprob('</s>', '</s>'))
        last_total_prob = [self.viterbi[i][len(observations) - 1] - self.tlprob(state, '</s>') for i, state in enumerate(self.states)]

        # TODO
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        # I got the minimum state index which will minimize the transition cost from that state to </s>. Thus
        # the backpointer process starts. Backpointer two dimensional array will take us to the start of the sentence
        # by pointing to the minimum cost state from a state.
        back_p = np.argmin(last_total_prob)
        tags = [self.states[back_p]]

        for i in range(len(observations)-1, 0, -1):
            tag = self.backpointer[back_p][i]
            # I appended the states in front each time because backtracking process is reversed.
            tags = [self.states[tag]] + tags
            back_p = tag
        return tags

    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42 
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        #raise NotImplementedError('HMM.get_viterbi_value')
        state_ind = self.states.index(state)
        return self.viterbi[state_ind][step] # fix me

    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step
        :type step: str
        :return: The state name to go back to at step-1
        :rtype: str
        """
        #raise NotImplementedError('HMM.get_backpointer_value')
        state_ind = self.states.index(state)
        return self.backpointer[state_ind][step] # fix me

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: your answer [max 280 chars]
    """
    #raise NotImplementedError('answer_question4b')

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [('Japan', 'NOUN'), (',', '.'), ('since', 'ADP'), ('1957', 'NUM'), (',', '.'), ('has', 'VERB'), 
    ('been', 'VERB'), ('``', '.'), ('voluntarily', 'ADV'), ("''", '.'), ('curbing', 'X'), ('exports', 'X'), ('of', 'ADP'), 
    ('textiles', 'NUM'), ('to', 'ADP'), ('the', 'DET'), ('U.S.', 'NOUN'), ('.', '.')]


    correct_sequence = [('Japan', 'NOUN'), (',', '.'), ('since', 'ADP'), ('1957', 'NUM'), (',', '.'), ('has', 'VERB'), 
    ('been', 'VERB'), ('``', '.'), ('voluntarily', 'ADV'), ("''", '.'), ('curbing', 'VERB'), ('exports', 'NOUN'), ('of', 'ADP'), 
    ('textiles', 'NUM'), ('to', 'ADP'), ('the', 'DET'), ('U.S.', 'NOUN'), ('.', '.')]
    # Why do you think the tagger tagged this example incorrectly?

    # curbing VERB (-20.29)  + "." "VERB" (-3.51)
    # curbing X (-12.74)  + "." "X" (-10.55)
    answer =  inspect.cleandoc("""\
    HMM that we build looks at the word in a bigram fashion: there might be less ambiguities/errors if the sentence
    is tagged in a trigram or more. The model tagged 'curbing' as X because it has a lower cost(emission + transition)
    compared to VERB.""")[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    #raise NotImplementedError('answer_question5')

    return inspect.cleandoc("""\
     Even though there will be 100% coverage of constructions, there might be unseen words because the training 
     set might not contain every possible word along with each of its possible tags. In this case, the emission 
     probabilities will be 0. To avoid this we will address the idea of smoothing to steal probability mass from 
     seen events and reallocate it to unseen events. It will always do better than the original parser on its own
     because it now allows unseen words.
     """)[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    #raise NotImplementedError('answer_question6')

    return inspect.cleandoc("""\
    It is because the Universal tagset is more generic to English: there are many specific
    tags in Brown Corpus which will affect tagging accuracy. Approximately 85 part-of-speech tags
    are used in Brown corpus, which means that it is hard to get 100% coverage on constructions.
    Some of the tags are very specific, which might cause 0 probabilities of some transitions.
    The construction data will be more sparse causing lower confidence in the predictions.
    """)[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5
    
    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal)-500 # fixme

    test_data_universal = tagged_sentences_universal[-test_size:] # fixme
    train_data_universal = tagged_sentences_universal[:train_size] # fixme

    if hashlib.md5(''.join(map(lambda x:x[0],train_data_universal[0]+train_data_universal[-1]+test_data_universal[0]+test_data_universal[-1])).encode('utf-8')).hexdigest()!='164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect, most of your answers will be wrong hereafter!!!'%(len(train_data_universal),len(test_data_universal)),file=sys.stderr)

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample=model.elprob('VERB','is')
    if not (type(e_sample)==float and e_sample<=0.0):
        print('elprob value (%s) must be a log probability'%e_sample,file=sys.stderr)

    t_sample=model.tlprob('VERB','VERB')
    if not (type(t_sample)==float and t_sample<=0.0):
           print('tlprob value (%s) must be a log probability'%t_sample,file=sys.stderr)

    if not (type(model.states)==list and \
            len(model.states)>0 and \
            type(model.states[0])==str):
        print('model.states value (%s) must be a non-empty list of strings'%model.states,file=sys.stderr)

    print('states: %s\n'%model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s) # fixme
    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)
    if not (type(b_sample)=='str' and b_sample in model.steps):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)


    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0
    counter = 0
    wrong = False
    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)
        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct += 1
            else:
                wrong = True
                incorrect += 1
        if wrong and counter <=10:
            print(s)
            counter += 1
            wrong = False
    accuracy = correct / (correct + incorrect)
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Print answers for 4b, 5 and 6
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nEmission log prob value for curbing being VERB:')
    print(model.elprob('VERB', 'curbing'))
    print('Emission log prob value for curbing being X:')
    print(model.elprob('X', 'curbing'))
    print('\nTransition log prob value for "." following a VERB:')
    print(model.tlprob('.', 'VERB'))
    print('Transition log prob value for "." following a X:')
    print(model.tlprob('.', 'X'))
    print('\nDiscussion of the difference:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])
    answer6=answer_question6()
    print('\nFor Q6:')
    print(answer6[:500])

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        import adrive2_embed
        from autodrive_embed import run, carefulBind
        with open("userErrs.txt","w") as errlog:
            run(globals(),answers,adrive2_embed.a2answers,errlog)
    else:
        answers()
