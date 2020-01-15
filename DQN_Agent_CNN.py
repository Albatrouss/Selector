# base construrct inspired by https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# bootstrap based on https://arxiv.org/pdf/1901.02219.pdf
# Author: Daniel Hämmerle

import random, numpy, time
from Logger import Logger
import keras
from keras.layers import Input, Dense, Flatten
from keras.models import Model

###hyperparameters
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
GAMMA = 0.99
LAMBDA = 0.001
SHARED_EXPERIENCE = 0.8


class Brain:
    def __init__(self, input_shape, action_count, head_count, name):
        self.logger = Logger(name=name, filename=name)
        self.name = name
        self.input_shape = input_shape
        self.action_count = action_count
        self.head_count = head_count
        self.total_model, self.models = self.create_model(self.head_count)

    def create_model(self, head_count):
        """ creates a sequential neural net with head_count heads as output layers"""
        init = init_head = "glorot_uniform"
        # create the input layer
        inputs = Input(shape=( self.input_shape))
        conv1 = keras.layers.Conv2D(input_shape=(self.input_shape), filters=32, kernel_size=8, strides=(4,4),
                                    data_format="channels_last", padding='valid',
                                    dilation_rate=(1, 1), activation=None, use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                    activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)
        conv2 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2,2), padding='valid',
                                    dilation_rate=(1, 1), activation=None, use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                    activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv1)
        conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='valid',
                                    dilation_rate=(1, 1), activation=None, use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                    activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv2)
        # create the layers that learn the game
        f1 = Flatten()(conv3)
        d1 = Dense(512, activation="relu", name="dense_1_shared", kernel_initializer=init)(f1)
        d2 = Dense(512, activation="relu", name="dense_2_shared", kernel_initializer=init)(d1)
        # create the heads that come on top of the gamelayers
        models = []
        heads = []
        for i in range(head_count):
            name = "head_{}".format(i)
            head = Dense(self.action_count, activation='linear', name=name, kernel_initializer=init_head)(d2)
            heads.append(head)
            model = Model(input=inputs, output=head, name=("head_{}".format(i)))
            model.compile(loss='mse', optimizer='adam')
            models.append(model)
        total_model = Model(input=inputs, output=heads, name="overall_modell")
        total_model.compile(loss='mse', optimizer='adam')
        return total_model, models

    def load_model(self, saved_model):
        """loads weights for the total_model into the model, therefore initializing all the individual head models"""
        self.total_model.load_weights(saved_model)

    def train_with_mask(self, x, y, mask, epoch=1, verbose=0):
        """fits a head with training data"""
        sample_weights = {"head_{}".format(x): w for x, w in enumerate(mask)}
        self.total_model.fit(x, y, sample_weight=sample_weights, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def predict_ensemble(self, state, mode='voting', std='normal', head_num=None):
        """
        return the average and the standard deviation of the predictions of all heads combines

        mode:   'average': returns the action based of the average of all predictions
                'voting': returns the action based of majority voting of all predicitons
                'single_head': returns prediction of a specific single head

        std:    'normal': returns the average of the standard deviations over all axes
                'square': returns the average of the squared standard deviations over all axes

        """
        predictions = []
        prediction = self.total_model.predict(state.reshape((1,) + self.input_shape))
        prediction = numpy.array(prediction).flatten()

        for i in range(0, self.head_count, 1):
            predictions.append(prediction[i * self.action_count: (i + 1) * self.action_count])

        stddev = None
        if std == 'squared':
            stddev = self.get_std_punished(predictions)
        if std == 'normal':
            stddev = self.get_std(predictions)
        if mode == 'average':
            return numpy.argmax(numpy.average(predictions, axis=0))
        if mode == 'voting':
            list_of_actions = [numpy.argmax(p) for p in predictions]
            majority_vote = max(set(list_of_actions), key=list_of_actions.count)
            return majority_vote, stddev
        if mode == 'single_head':
            return numpy.argmax(predictions[head_num]), stddev

    def predict_one(self, state, head):
        a = self.models[head].predict(state.reshape((1,) + self.input_shape)).flatten()
        return a

    def predict(self, state, head):
        """returns predictions of one head for an array of states"""
        return self.models[head].predict(state)

    def get_std(self, predictions):
        """returns the average standard deviation of the predictions"""
        return numpy.average(numpy.std(predictions, axis=0))

    def get_std_punished(self, predictions):
        """ punishes higher standard deviations by squaring them before averaging them out"""
        return numpy.average([x ** 2 for x in numpy.std(predictions, axis=0)])


'''
    def train(self, x, y, head, epoch=1, verbose=0):
        """fits a head with training data"""
        self.models[head].fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)
'''


class Memory:

    def __init__(self, memory_capacity, head_count):
        self.memory_capacity = memory_capacity
        self.head_count = head_count
        random.seed(int(time.time()))
        self.samples = []

    def add(self, sample):
        """generates a random mask and adds the sample to the memories belonging to the heads selected by the mask"""
        mask = numpy.random.choice([0.00000000001, 1], size=self.head_count,
                                   p=[1 - SHARED_EXPERIENCE, SHARED_EXPERIENCE])
        self.samples.append((sample, mask))

    def sample(self, n):
        return random.sample(self.samples, min(n, len(self.samples)))


class Agent:

    def __init__(self, input_dim, action_count, head_count, name):
        self.name = name
        self.input_dim = input_dim
        self.action_count = action_count
        self.head_count = head_count
        self.brain = Brain(input_dim, action_count, head_count, name)
        self.memory = Memory(MEMORY_CAPACITY, head_count)

    def load_model(self, model):
        """loads a model into the brain"""
        self.brain.load_model(model)

    def act_on_head(self, state, head_num):
        """uses a single head to get a prediction"""
        p = self.brain.predict_one(state, head_num)
        return numpy.argmax(p)

    def act(self, state):
        """gets a prediction from all the heads, returns strongest one as well as the standard deviation amongst the
        heads """
        return self.brain.predict_ensemble(state)

    def observe(self, sample):
        """adds a sample to memory"""
        self.memory.add(sample)

    def replay(self):
        """retrains the brain on past memories"""
        batch = self.memory.sample(BATCH_SIZE)
        batch_length = len(batch)
        if batch_length == 0:
            return None

        head_num = numpy.random.choice(range(self.head_count))
        no_state = numpy.zeros(self.input_dim)
        states = numpy.array([sample[0] for sample, mask in batch])  # starting states of actions in the batch
        ending_states = numpy.array([(no_state if sample[3] is None else sample[3]) for sample, mask in batch])
        # ending states of actions in batch, unless finalstate, then 0

        predictions = self.brain.predict(states, head=head_num)  # predicted reward at start
        predictions_of_end = self.brain.predict(ending_states, head=head_num)  # predicted reward at end
        x = numpy.zeros(((batch_length,) + self.input_dim))
        y = numpy.zeros((batch_length, self.action_count))
        weights = [mask for sample, mask in batch]
        for i in range(batch_length):
            sample, mask = batch[i]
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            end_state = sample[3]

            t = predictions[i]
            if end_state is None:
                t[action] = reward
            else:
                t[action] = reward + numpy.amax(predictions_of_end[i]) * GAMMA

            x[i] = state

            y[i] = t

        weights = numpy.array(weights).T
        y_new = [y for i in range(self.head_count)]
        self.brain.train_with_mask(x, y_new, weights)
