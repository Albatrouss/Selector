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
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(output_dim=128, activation='relu', input_dim=self.state_count))
        model.add(Dense(output_dim=128, activation='relu', input_dim=128))
        model.add(Dense(output_dim=self.action_count, activation='linear'))
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        print(model.summary())
        return model

    def load_model(self, toload):
        self.model.load_weights(toload)

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

    def predict(self, state):
        return self.model.predict(state)

    def predict_one(self, state):
        return self.model.predict(state.reshape(1, self.state_count)).flatten()


class Memory:
    samples = []  # store samples in list

    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity

    def add(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.memory_capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count
        self.brain = Brain(state_count, action_count)
        self.memory = Memory(MEMORY_CAPACITY)

    def loadModel(self, model, max_epsilon):
        self.brain.load_model(model)
        self.epsilon = max_epsilon

    def act(self, state):
        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        else:
            return numpy.argmax(self.brain.predict_one(state))

    def observe(self, sample):
        self.memory.add(sample)
        # add a step and decrease epsilon
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batch_length = len(batch)

        no_state = numpy.zeros(self.state_count)

        states = numpy.array([o[0] for o in batch])  # starting states of actions in the batch
        end_states = numpy.array([(no_state if o[3] is None else o[3]) for o in
                                  batch])  # ending states of actions in batch, unless finalstate, then  0

        predictions = self.brain.predict(states)  # rewards that were predicted for the starting states
        predictions_of_end = self.brain.predict(end_states)  # reward that were predicted for the ending states

        x = numpy.zeros((batch_length, self.state_count))
        y = numpy.zeros((batch_length, self.action_count))

        for i in range(batch_length):
            o = batch[i]
            state = o[0]
            action = o[1]
            reward = o[2]
            ending_state = o[3]

            t = predictions[i]
            if ending_state is None:
                t[action] = reward
            else:
                t[action] = reward + numpy.amax(predictions_of_end[i]) * GAMMA

            x[i] = state
            y[i] = t

        self.brain.train(x, y)
