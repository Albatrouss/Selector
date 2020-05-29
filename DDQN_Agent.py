# base construrct inspired by https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# bootstrap based on https://arxiv.org/pdf/1901.02219.pdf
# Author: Daniel Hämmerle

import random, numpy, time
import sys

import keras
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import *
import pickle
import datetime


###hyperparameters


class Brain:
    def __init__(self, input_shape, action_count, head_count, batch_size, alpha=0.00025, mode="conv"):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.action_count = action_count
        self.head_count = head_count
        self.alpha = alpha
        self.mode = mode
        self.total_model, self.models = self.create_model(self.head_count)

    def create_model(self, head_count):
        """ creates a sequential neural net with head_count heads as output layers

            mode determines the shape and nature of the neural network:
            "conv"(default): convolutional network into 1 dense into heads
            "simple" 2 dense layers into heads

        """
        init = init_head = "glorot_uniform"
        # create the input layer
        if self.mode == "conv" or self.mode == "conv2":
            inputs = Input(shape=(self.input_shape))
            conv1 = keras.layers.Conv2D(input_shape=(self.input_shape), filters=32, kernel_size=8, strides=(4, 4),
                                        data_format="channels_last", padding='valid',
                                        dilation_rate=(1, 1), activation=None, use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)
            conv2 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='valid',
                                        dilation_rate=(1, 1), activation=None, use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv1)
            conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid',
                                        dilation_rate=(1, 1), activation=None, use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(conv2)
            f1 = Flatten()(conv3)
            dfinal = Dense(512, activation="relu", name="dense_1_shared", kernel_initializer=init)(f1)
        if self.mode == "simple":
            inputs = Input(shape=(self.input_shape[0],))
            d1 = Dense(64, activation="relu", name="dense_1_shared", kernel_initializer=init)(inputs)
            dfinal = Dense(64, activation="relu", name="dense_2_shared", kernel_initializer=init)(d1)
        # add heads on top
        models = []
        heads = []
        for i in range(head_count):
            if self.mode == "conv2":
                name = "head_{}".format(i)
                head_dense = Dense(64, activation="relu", name="dense_{}_head".format(i), kernel_initializer=init)(
                    dfinal)
                head = Dense(self.action_count, activation='linear', name=name, kernel_initializer=init_head)(
                    head_dense)
                heads.append(head)
            else:
                name = "head_{}".format(i)
                head = Dense(self.action_count, activation='linear', name=name, kernel_initializer=init_head)(dfinal)
                heads.append(head)
            model = Model(input=inputs, output=head, name=("head_{}".format(i)))
            model.compile(loss='mse', optimizer='adam')
            models.append(model)
        total_model = Model(input=inputs, output=heads, name="overall_modell")
        # opt = RMSprop(learning_rate=self.alpha, )
        my_adam = Adam(learning_rate=self.alpha)
        total_model.compile(loss='mse', optimizer=my_adam)
        return total_model, models

    def train(self, x, y, mask, epoch=1, verbose=0):
        """fits a head with training data"""
        sample_weights = {"head_{}".format(x): w for x, w in enumerate(mask)}
        self.total_model.fit(x, y, sample_weight=sample_weights, batch_size=self.batch_size, epochs=epoch,
                             verbose=verbose)

    def predict_single_state(self, state, mode='voting', std='both', head_num=None):
        """
        return the average and the standard deviation of the predictions of all heads combines

        mode:   'average': returns the action based of the average of all predictions
                'voting': returns the action based of majority voting of all predicitons
                'single_head': returns prediction of a specific single head

        std:    'normal': returns the average of the standard deviations over all axes
                'square': returns the average of the squared standard deviations over all axes
                'both': returns a tuple with (square, normal)

        """
        predictions = []
        prediction = self.total_model.predict(state.reshape((1,) + self.input_shape))
        prediction = numpy.array(prediction).flatten()

        for i in range(0, self.head_count, 1):
            predictions.append(prediction[i * self.action_count: (i + 1) * self.action_count])

        stddev = 0
        if std == 'squared':
            stddev = self.get_std_punished(predictions)
        if std == 'normal':
            stddev = self.get_std(predictions)
        if std == 'both':
            stddev = (self.get_std_punished(predictions), self.get_std(predictions))
        if mode == 'average':
            return numpy.argmax(numpy.average(predictions, axis=0)), stddev
        if mode == 'voting':
            list_of_actions = [numpy.argmax(p) for p in predictions]
            majority_vote = max(set(list_of_actions), key=list_of_actions.count)
            return majority_vote, stddev
        if mode == 'single_head':
            return numpy.argmax(predictions[head_num]), stddev

    def predict_multiple_states(self, state, head):
        """returns predictions of one head for an array of states, used in replay"""
        return self.models[head].predict(state)

    def get_std(self, predictions):
        """returns the average standard deviation of the predictions"""
        return numpy.average(numpy.std(predictions, axis=0))

    def get_std_punished(self, predictions):
        """ punishes higher standard deviations by squaring them before averaging them out"""
        return numpy.average([x ** 2 for x in numpy.std(predictions, axis=0)])


class Memory:

    def __init__(self, memory_capacity, head_count, shared_exp):
        self.memory_capacity = memory_capacity
        self.head_count = head_count
        self.shared_exp = shared_exp
        random.seed(int(time.time()))
        self.samples = []

    def add(self, sample):
        """generates a random mask and adds the sample to the memories belonging to the heads selected by the mask"""
        if len(self.samples) > self.memory_capacity:
            self.samples.pop(0)
        zero = sys.float_info.min
        mask = numpy.random.choice([zero, 1], size=self.head_count, p=[1 - self.shared_exp, self.shared_exp])
        self.samples.append((sample, mask))

    def sample(self, n):
        return random.sample(self.samples, min(n, len(self.samples)))

    def write_to_file(self, filename, mem_multiplier):
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump(random.sample(self.samples, min(len(self.samples),
                                                        int(self.memory_capacity * mem_multiplier))), fp)

    def read_from_file(self, filename):
        with open(filename, "rb") as fp:  # Unpickling
            self.samples = pickle.load(fp)


class Agent:

    def __init__(self, input_dim, action_count, head_count, name, steps_before_learning=10000, epsilon=1,
                 epsilon_decay=0.00000099, epsilon_min=0.01, batch_size=32, target_update_intervall=10000,
                 memory_capacity=1000000, alpha=0.00025, shared_exp=0.8, gamma=0.99, mode="conv",
                 save_interval=1000000, save_big_path="/big/h/haemmerle/", save_memory_multiplier=0.1):
        self.save_interval = save_interval
        self.save_memory_multiplier = save_memory_multiplier
        self.save_big_path = save_big_path
        self.shared_exp = shared_exp
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.train = True
        self.steps = 0
        self.steps_before_learning = steps_before_learning
        self.name = name
        self.input_dim = input_dim
        self.action_count = action_count
        self.head_count = head_count
        self.batch_size = batch_size
        self.target_update_intervall = target_update_intervall
        self.online_network = Brain(input_dim, action_count, head_count, alpha=alpha, mode=mode,
                                    batch_size=self.batch_size)
        self.target_network = Brain(input_dim, action_count, head_count, alpha=alpha, mode=mode,
                                    batch_size=self.batch_size)
        self.memory = Memory(memory_capacity, head_count, shared_exp)

    def load_model(self):
        """loads a saved model including weights, memory, steps and epsilon"""
        self.online_network.total_model.load_weights("models/{}.h5".format(self.name))
        self.target_network.total_model.load_weights("models/{}.h5".format(self.name))
        # load memory
        self.memory.samples = []
        self.memory.read_from_file("{}{}.mem".format(self.save_big_path, self.name))
        # load steps and epsilon
        mylogfile = open("models/{}.csv".format(self.name), 'r')
        firstRow = mylogfile.readline()
        fieldnames = firstRow.strip('\n').split(";")
        self.steps = int(fieldnames[0])
        self.epsilon = float(fieldnames[1])

    def save_agent(self):
        """saves a model, including weights, memory, steps and epsilon"""
        # save model: name.h5
        self.online_network.total_model.save("models/{}.h5".format(self.name))
        # save memory: name.mem
        self.memory.write_to_file("{}{}.mem".format(self.save_big_path, self.name), self.save_memory_multiplier)
        # save steps and epsilon: name.csv
        timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
        string_data = str(self.steps) + ";" + str(self.epsilon) + ";" + timestamp
        mylogfile = open("models/{}.csv".format(self.name), 'w+')
        mylogfile.write(string_data)
        mylogfile.close()

    def act_on_head(self, state, head_num):
        """uses a single head to get a prediction"""
        if self.train:
            self.steps += 1
            if self.steps % self.target_update_intervall == 0:
                self.update_target_net()
            if self.steps % self.save_interval == 0:
                self.save_agent()
            if random.random() < self.epsilon:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                return random.randint(0, self.action_count - 1), -1
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            return self.online_network.predict_single_state(state, mode="single_head", head_num=head_num)

        else:
            # evaluation: majority voting, ignoring the acting on a single head
            return self.online_network.predict_single_state(state)

    def act(self, state):
        """gets a prediction from all the heads, returns strongest one as well as the standard deviation amongst the
        heads """
        if self.train:
            self.steps += 1
            if self.steps % self.target_update_intervall == 0:
                self.update_target_net()
            if self.steps % self.save_interval == 0:
                self.save_agent()
            if random.random() < self.epsilon:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                return random.randint(0, self.action_count - 1), -1
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            return self.online_network.predict_single_state(state)
        else:
            # evaluation: majority voting
            return self.online_network.predict_single_state(state)

    def observe(self, sample):
        """adds a sample to memory"""
        self.memory.add(sample)

    def update_target_net(self):
        self.target_network.total_model.set_weights(self.online_network.total_model.get_weights())

    def replay(self):
        """retrains the brain on past memories"""
        batch = self.memory.sample(self.batch_size)
        batch_length = len(batch)
        if batch_length == 0 or self.steps < self.steps_before_learning:
            return None

        no_state = numpy.zeros(self.input_dim)
        states = numpy.array([sample[0] for sample, mask in batch])  # starting states of actions in the batch
        ending_states = numpy.array([(no_state if sample[3] is None else sample[3]) for sample, mask in batch])
        # ending states of actions in batch, unless finalstate, then 0

        predictions = numpy.hstack(self.online_network.total_model.predict(states))
        predictions_end = numpy.hstack(self.online_network.total_model.predict(ending_states))
        predictions_target = numpy.hstack(self.target_network.total_model.predict(ending_states))
        x = numpy.zeros(((batch_length,) + self.input_dim))
        y = numpy.zeros((batch_length, self.action_count * self.head_count))

        for i in range(batch_length):
            sample, mask = batch[i]
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            end_state = sample[3]

            t = predictions[i]
            t_e = predictions_end[i]
            if end_state is None:
                for j in range(self.head_count):
                    if mask[j] == 1:
                        t[action + j * self.action_count] = reward
            else:
                target_pred = predictions_target[i]
                for j in range(self.head_count):
                    if mask[j] == 1:
                        # make temporary t of just this head
                        t_tmp = t[j * self.action_count:(j + 1) * self.action_count]
                        t_e_tmp = t_e[j * self.action_count:(j + 1) * self.action_count]
                        # make temporary predictions for just this head
                        target_pred_tmp = target_pred[j * self.action_count:(j + 1) * self.action_count]
                        # adjust the reward for the action taken to be the reward acc. to ddqn
                        # print("numpy.amax:{}".format(numpy.argmax(t_tmp)))
                        t_tmp[action] = reward + self.gamma * target_pred_tmp[numpy.argmax(t_e_tmp)]
                        # write it back to the t
                        t[j * self.action_count:(j + 1) * self.action_count] = t_tmp
            x[i] = state
            y[i] = t
        # split into arrays for each head
        y = numpy.hsplit(y, self.head_count)
        weights = [m for o, m in batch]
        weights = numpy.array(weights).T
        # self.online_network.train(x, y)
        self.online_network.train(x, y, weights)
