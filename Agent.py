# following the tutorial at https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# Author: Daniel Hämmerle

import random, numpy, math, gym
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *

###hyperparameters
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001

class Brain:
    def __init__(self, statecount, actioncount):
        self.statecount = statecount
        self.actioncount = actioncount
        self.model = self._createModel()

    def _createModel(self):
        model = Sequential()
        model.add(Dense(output_dim=128, activation='relu', input_dim=self.statecount))
        model.add(Dense(output_dim=128, activation='relu', input_dim=128))
        model.add(Dense(output_dim=self.actioncount, activation='linear'))
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        print(model.summary())
        return model

    def _loadModel(self, toload):
        self.model.load_weights(toload)

    def train(self, x, y, epoch=1, verbose = 0):
        self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

    def predict(self, state):
        return self.model.predict(state)

    def predict_one(self, state):
        return self.model.predict(state.reshape(1, self.statecount)).flatten()


class Memory:
    samples = [] #store samples in list

    def __init__(self, memorycapacity):
        self.memorycapacity = memorycapacity

    def add(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.memorycapacity:
            self.samples.pop(0)
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples,n)

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, statecount, actioncount):
        self.statecount = statecount
        self.actioncount = actioncount
        self.brain = Brain(statecount, actioncount)
        self.memory = Memory(MEMORY_CAPACITY)

    def loadModel(self, model, maxepsilon):
        self.brain._loadModel(model)
        self.epsilon = maxepsilon

    def act(self, state):
        #Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.actioncount -1)
        else:
            return numpy.argmax(self.brain.predict_one(state))

    def observe(self, sample):
        self.memory.add(sample)
        #add a step and decrease epsilon
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchlength = len(batch)

        nostate = numpy.zeros(self.statecount)

        states = numpy.array([ o[0] for o in batch]) # starting states of actions in the batch
        endstates = numpy.array([(nostate if o[3] is None else o[3]) for o in batch ]) # ending states of actions in batch, unless finalstate, then  0

        predictions = self.brain.predict(states) #rewards that were predicted for the starting states
        predictionsofend = self.brain.predict(endstates) #reward that were predicted for the ending states

        x = numpy.zeros((batchlength, self.statecount))
        y = numpy.zeros((batchlength, self.actioncount))

        for i in range(batchlength):
            o = batch[i]
            state = o[0]
            action = o[1]
            reward = o[2]
            endstate = o[3]

            t = predictions[i]
            if endstate is None:
                t[action] = reward
            else:
                t[action] = reward + numpy.amax(predictionsofend[i]) * GAMMA

            x[i] = state
            y[i] = t

        self.brain.train(x, y)