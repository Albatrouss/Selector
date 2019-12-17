# base construrct inspired by https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# bootstrap based on https://arxiv.org/pdf/1901.02219.pdf
# Author: Daniel Hämmerle

import random, numpy, time
from Logger import Logger
import keras
from keras.layers import Input, Dense
from keras.models import Model

###hyperparameters
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LAMBDA = 0.001
SHARED_EXPERIENCE = 0.7


class Brain:
    def __init__(self, state_count, action_count, head_count, name):
        self.logger = Logger(name=name, filename=name)
        self.name = name
        self.state_count = state_count
        self.action_count = action_count
        self.head_count = head_count
        self.total_model, self.models = self.create_model(self.head_count)

    def create_model(self, head_count):
        """ creates a sequential neural net with head_count heads as output layers"""
        # create the input layer
        inputs = Input(shape=(self.state_count,))
        # create the layers that learn the gane
        d1 = Dense(64, activation="relu", name="dense_1_shared",
                   kernel_initializer=keras.initializers.glorot_uniform(seed=int(time.time())))(inputs)
        d2 = Dense(64, activation="relu", name="dense_2_shared",
                   kernel_initializer=keras.initializers.glorot_uniform(seed=None))(d1)
        # create the heads that come on top of the gamelayers
        models = []
        heads = []
        for i in range(head_count):
            name = "head_{}".format(i)
            head = Dense(self.action_count, activation='relu', name=name,
                         kernel_initializer=keras.initializers.glorot_uniform(seed=None))(d2)
            heads.append(head)
            model = Model(input=inputs, output=head, name=("headmodel: {}".format(str(i))))
            model.compile(loss='mse', optimizer='adam')
            models.append(model)
        total_model = Model(input=inputs, output=heads, name="overall_modell")
        total_model.compile(loss='mse', optimizer='adam')
        return total_model, models

    def load_model(self, saved_model):
        """loads weights for the total_model into the model, therefore initializing all the individual head models"""
        self.total_model.load_weights(saved_model)

    def train(self, x, y, head, epoch=1, verbose=0):
        """fits a head with training data"""
        self.models[head].fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def predict_ensemble(self, state):
        """return the average and the standard deviation of the predictions of all heads combines """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(state.reshape(1, self.state_count)).flatten())
        newstd = self.get_std_punished(predictions)
        pstd = self.get_std(predictions)
        self.logger.add_data([newstd, pstd, numpy.average(predictions, axis=0)])
        return numpy.average(predictions, axis=0), newstd

    def predict_one(self, state, head):
        return self.models[head].predict(state.reshape(1, self.state_count)).flatten()

    def predict(self, state, head):
        """returns predictions of one head for an array of states"""
        return self.models[head].predict(state)

    def get_std(self, predictions):
        """returns the average standard deviation of the predictions"""
        return numpy.average(numpy.std(predictions, axis=0))

    def get_std_punished(self, predictions):
        """ punishes higher standard deviations by squaring them before averaging them out"""
        return numpy.average([x ** 2 for x in numpy.std(predictions, axis=0)])


class Memory:

    def __init__(self, memory_capacity, head_count):
        self.memory_capacity = memory_capacity
        self.head_count = head_count
        random.seed(42)  # for reproducibility
        self.samples = [[] for i in range(self.head_count)]

    def add(self, sample):
        """generates a random mask and adds the sample to the memories belonging to the heads selected by the mask"""
        mask = numpy.random.choice([0, 1], size=self.head_count, p=[1 - SHARED_EXPERIENCE, SHARED_EXPERIENCE])
        for m in range(len(mask)):
            if mask[m] == 1:
                self.samples[m].append(sample)
                if len(self.samples[m]) > self.memory_capacity:
                    self.samples[m].pop(0)

    def sample(self, n):
        """samples the heads memories and returns a batch for each one of them in the form of a list of lists"""
        sampled = [[] for i in range(self.head_count)]
        for i in range(self.head_count):
            sampled[i] = random.sample(self.samples[i], min(n, len(self.samples[i])))
        return sampled


class Agent:

    def __init__(self, state_count, action_count, head_count, name):
        self.name = name
        self.state_count = state_count
        self.action_count = action_count
        self.head_count = head_count
        self.brain = Brain(state_count, action_count, head_count, name)
        self.memory = Memory(MEMORY_CAPACITY, head_count)

    def load_model(self, model):
        """loads a model into the brain"""
        self.brain.load_model(model)

    def act_on_head(self, state, head_num):
        """uses a single head to get a prediction"""
        return numpy.argmax(self.brain.predict_one(state, head_num))

    def act(self, state):
        """gets a prediction from all the heads, returns strongest one as well as the standard deviation amongst the
        heads """
        p, std = self.brain.predict_ensemble(state)
        return numpy.argmax(p), std

    def observe(self, sample):
        """adds a sample to memory"""
        self.memory.add(sample)

    def replay(self):
        """retrains the brain on past memories"""
        batches = self.memory.sample(BATCH_SIZE)
        for head_num in range(len(batches)):

            batch = batches[head_num]
            batch_length = len(batch)

            if batch_length == 0:
                break

            no_state = numpy.zeros(self.state_count)

            states = numpy.array([o[0] for o in batch])  # starting states of actions in the batch
            ending_states = numpy.array([(no_state if o[3] is None else o[3]) for o in
                                         batch])  # ending states of actions in batch, unless finalstate, then  0
            predictions = self.brain.predict(states,
                                             head=head_num)  # rewards that were predicted for the starting states
            predictions_of_end = self.brain.predict(ending_states,
                                                    head=head_num)  # reward that were predicted for the ending states

            x = numpy.zeros((batch_length, self.state_count))
            y = numpy.zeros((batch_length, self.action_count))

            for i in range(batch_length):
                o = batch[i]
                state = o[0]
                action = o[1]
                reward = o[2]
                end_state = o[3]

                t = predictions[i]
                if end_state is None:
                    t[action] = reward
                else:
                    t[action] = reward + numpy.amax(predictions_of_end[i]) * GAMMA

                x[i] = state
                y[i] = t

            self.brain.train(x, y, head=head_num)
