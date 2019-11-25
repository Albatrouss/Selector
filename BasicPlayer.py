# following the tutorial at https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# Author: Daniel Hämmerle

import random, numpy, math, gym
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from sklearn.neighbors import KNeighborsClassifier
import pandas
import lunar_lander as ll
import Selector

###hyperparameters
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99

startingstatefile = ["savefile1.csv", "savefile2.csv"]
models = ["ll1.h5", "ll2.h5"]
modelselected = 0
MAX_EPSILON =  0.01 #1 for training 0.01 for execution
MIN_EPSILON = 0.01
LAMBDA = 0.001

class Brain:
    def __init__(self, statecount, actioncount):
        self.statecount = statecount
        self.actioncount = actioncount
        self.model = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=128, activation='relu', input_dim=statecount))
        model.add(Dense(output_dim=128, activation='relu', input_dim=statecount))
        model.add(Dense(output_dim=actioncount, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        # if you want to load weights from previous training:
        #model.load_weights(models[modelselected])
        #print("Model loaded:" + models[modelselected])
        return model

    def _loadModel(self, toload):
        self.model.load_weights(toload)
        print("Model loaded " + toload)

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

    def __init__(self, statecount, actioncount, selector):
        self.statecount = statecount
        self.actioncount = actioncount
        self.selector = selector
        self.brain = Brain(statecount, actioncount)
        self.memory = Memory(MEMORY_CAPACITY)

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

        self.brain.train(x,y)


class Environment:
    def __init__(self, env):
        #self.problem = problem
        self.environment = env #gym.make(problem)

    def run(self, agent):
        state = self.environment.reset()
        Reward = 0
        ## hier muss ein modell ausgewählt werden, weil wir zum ersten mal den state haben.
        if agent.selector.predictOne(state)[0] == "EINS":
            print("success : predicted 1")
            agent.brain._loadModel("ll1.h5")
        if agent.selector.predictOne(state)[0] == "ZWEI":
            print("success : predicted2")
            agent.brain._loadModel("ll2.h5")
        #agent.brain._loadModel(modelselected)


        #startingstates.append(state)
        while True:
            self.environment.render()
            action = agent.act(state)

            newstate, currentreward, done, info = self.environment.step(action)

            if done:
                newstate = None

            agent.observe((state, action, currentreward, newstate))
            agent.replay()

            state = newstate
            Reward += currentreward

            if done:
                break
        print("Reward achieved: ", Reward)



############################## MAIN
#PROBLEM = 'CartPole-v0'
#my_environment = Environment(PROBLEM)
my_problem = ll.LunarLanderModable()
my_problem.seed(42)
#
env_params1 = {
     "multiagent_compatibility": False,
     "helipad_y_ranges": [(0, 3)],  # train
     "helipad_x_ranges": [(1, 3)],  # train
}
env_params2 = {
     "multiagent_compatibility": False,
     "helipad_y_ranges": [(5, 8)],  # train
     "helipad_x_ranges": [(8, 12)],  # train
}
modelselected = random.randint(0, 1)
if modelselected == 0:
    my_problem.load_config(env_params1)
    print("Environment loaded: env_params1")
if modelselected == 1:
    my_problem.load_config(env_params2)
    print("Environment loaded: env_params2")

my_environment = Environment(my_problem)

statecount = my_environment.environment.observation_space.shape[0]
actioncount = my_environment.environment.action_space.n

myselector = Selector.Selector(["taggeddata.csv"])

my_agent = Agent(statecount, actioncount, myselector)
startingstates = []
i = 0
try:
    while i<10:
        modelselected = random.randint(0, 1)
        if modelselected == 0:
            my_problem.load_config(env_params1)
            print("Environment loaded: env_params1")
        if modelselected == 1:
            my_problem.load_config(env_params2)
            print("Environment loaded: env_params2")

        my_environment.run(my_agent)

        i = i + 1
        print(i)
finally:
    #my_agent.brain.model.save(models[modelselected])
    #print("Model saved as " + models[modelselected])
    #for s in startingstates:
     print("xxx")
    #numpy.savetxt(startingstatefile[modelselected], startingstates,delimiter=";")
    #print("saved to " + startingstatefile[modelselected])