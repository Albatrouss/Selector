import random, numpy, math, gym
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from sklearn.neighbors import KNeighborsClassifier
import pandas
import lunar_lander as ll
from Environment import Environment
from Agent import Agent
from Selector import Selector
from gym import wrappers

############################################################ Main
#---------------define Environments-------------------------
my_problem1 = ll.LunarLanderModable()
my_problem1.seed(42)
my_problem2 = ll.LunarLanderModable()
my_problem2.seed(42)
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
my_problem1.load_config(env_params1)
my_environment1 = Environment(my_problem1)
my_problem2.load_config(env_params2)
my_environment2 = Environment(my_problem2)

#------------------------init Agents ---------------------

statecount1 = my_environment1.environment.observation_space.shape[0]
actioncount1 = my_environment1.environment.action_space.n

statecount2 = my_environment2.environment.observation_space.shape[0]
actioncount2 = my_environment2.environment.action_space.n

Agent1 = Agent(statecount1, actioncount1)
#Agent1.loadModel("ll1.h5",1)
Agent2 = Agent(statecount2, actioncount2)
#Agent2.loadModel("ll2.h5",1)

#------------------------init Selector ------------------

selector = Selector([Agent1, Agent2], ["data1.csv","data2.csv"])

#---------------------------run-----------------------------------

i = 0
verbose = True
train = True
try:
    while i<500:
          env = random.randint(0, 1)
          if env == 0:
             my_environment = my_environment1
             print("Environment loaded: env_params1")
          if env == 1:
             my_environment = my_environment2
             print("Environment loaded: env_params2")
          my_environment.run(selector, train, verbose, render=False)
          i = i + 1
          print(i)
finally:
     if train:
          Agent1.brain.model.save("ll1.2.h5")
          Agent2.brain.model.save("ll2.2.h5")
     print("Savedanddone")