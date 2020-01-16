import datetime
import random
import time
import sys

import gym

import lunar_lander as ll
from Environment import Environment
from DQN_Agent_CNN import Agent
from Uncertainty_Selector import Selector
from Logger import Logger

############################################################ Main
# ------------------------define modes----------------------
verbose = True
render = False
train = False
train_episodes = 500000
test_episodes = 100
head_count = 10
name = sys.argv[1]

# ---------------define Environments-------------------------

env = Environment(gym.make("procgen:procgen-chaser-v0", distribution_mode="easy"))
environments=[env]
# ------------------------init Agents ---------------------

input_dims = env.environment.observation_space.shape
print(input_dims)
action_count = env.environment.action_space.n


agent1 = Agent(input_dims, action_count, head_count,
               name="1")  # "Agent1_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
try:
    agent1.brain.total_model.load_weights("models/chaser{}.h5".format(name))
except:
    print("no file")

'''if not train:
    print("loading agents")
    agent1.brain.total_model.load_weights(
        "models/Environment1_500episodes.h5")
    agent2.brain.total_model.load_weights(
        "models/Environment2_500episodes.h5")'''

# ------------------------init Selectors ------------------

selector_specific = Selector([agent1])


# ---------------------------run-----------------------------------
def savemodel(agent,  name):
    agent.brain.total_model.save(
        "models/chaser{}.h5".format(name))  # {}Trained:{}_onEnv:{}_time:{}.h5".format(name,train_episodes,i,timestamp))

def train_test(train_episodes, test_episodes, selectorspecific, selectorsingle, environments, name, render):
    # train
    train_test_logger = Logger(name=name, filename="test_train_log_chaser_{}".format(name))
    train_test_logger.add_data(["name", "entry_number", "selected or train", "reward", "standard deviation","correctly_selected(opt)"])
    model_logger = Logger(name="modellogger", filename="modellogger_{}".format(name))
    entrynr = 0
    try:
        # multi agents
        for i, agent in enumerate(selectorspecific.agents):
            my_environment = environments[i]
            selectorspecific.training(i)
            for i in range(train_episodes):
                selected, reward, std = my_environment.run(selectorspecific, train=True, verbose=False, render=render)
                train_test_logger.add_data(
                    [str(name), str(entrynr), "train: {}".format(selected), str(reward), str(std)])
                entrynr += 1
                if entrynr % 100 == 0:
                    savemodel(agent, name)
                    model_logger.add_data(["episodes:", str(entrynr)])
        # test multi agents:
        selectorspecific.train = False
        for i in range(test_episodes):
            env_num, my_environment = random.choice(list(enumerate(environments)))
            selected, reward, std = my_environment.run(selectorspecific, train=False, verbose=False, render=render)
            train_test_logger.add_data(
                [str(name), str(entrynr), "test:specific:{}/{}".format(selected, env_num), str(reward), str(std),
                 str(selected == env_num)])
            entrynr += 1
            print(entrynr)
        # single agent
        '''
        for i in range(train_episodes):
            my_environment = random.choice(environments)
            _, reward, std = my_environment.run(selectorsingle, train=True, verbose=False, render=render)
            train_test_logger.add_data([str(name), str(entrynr), "train:generic", str(reward), str(std)])
            entrynr += 1
            print(entrynr)
        # test it
        for i in range(test_episodes):
            my_environment = random.choice(environments)
            _, reward, std = my_environment.run(selectorsingle, train=False, verbose=False, render=render)
            train_test_logger.add_data(
                [str(name), str(entrynr), "test:generic{}".format(train_episodes), str(reward), str(std)])
            entrynr += 1
            print(entrynr)
        # train it some more to match total episodes
        for i in range(train_episodes * (len(selectorspecific.agents) - 1)):
            my_environment = random.choice(environments)
            _, reward, std = my_environment.run(selectorsingle, train=True, verbose=False, render=render)
            train_test_logger.add_data([str(name), str(entrynr), "train:generic", str(reward), str(std)])
            entrynr += 1
        # test it again
        for i in range(test_episodes):
            my_environment = random.choice(environments)
            _, reward, std = my_environment.run(selectorsingle, train=False, verbose=False, render=render)
            train_test_logger.add_data(
                [str(name), str(entrynr), "test:generic{}".format(train_episodes * len(selectorspecific.agents)),
                 str(reward), str(std)])
            entrynr += 1
            print(entrynr)
        '''
    finally:
        for i, agent in enumerate(selectorspecific.agents):
            timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
            agent.brain.total_model.save("models/chaser.h5")#{}Trained:{}_onEnv:{}_time:{}.h5".format(name,train_episodes,i,timestamp))
        '''for i, agent in enumerate(selectorsingle.agents):
            timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
            agent.brain.total_model.save("models/{}Trained:{}_onAllEnv_time:{}.h5".format(name,train_episodes,timestamp))
            '''
        print("Done")


train_test(train_episodes, test_episodes, selector_specific, selector_specific, environments, name, render=render)
