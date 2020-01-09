import datetime
import random
import time
import sys

import lunar_lander as ll
from Environment import Environment
from Uncertainty_Agent import Agent
from Uncertainty_Selector import Selector
from Logger import Logger

############################################################ Main
# ------------------------define modes----------------------
verbose = True
render = True
train = False
train_episodes = 500
test_episodes = 100
head_count = 10
name = sys.argv[1]
#print(name)

# ---------------------define Loggers ------------------------------------------------------------
#main_logger = Logger(name="Main_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
#main_logger.add_data(["Episode_num; Environment_num;Agent_num;Head_num; Reward; Timestamp"])
#train_logger1 = Logger(name="training1", filename="1_outcome.csv")
#train_logger2 = Logger(name="training2", filename="2_outcome.csv")
# ---------------define Environments-------------------------
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
environments = [my_environment1, my_environment2]

# ------------------------init Agents ---------------------

state_count = my_environment1.environment.observation_space.shape[0]
action_count = my_environment1.environment.action_space.n


agent1 = Agent(state_count, action_count, head_count,
               name="1")  # "Agent1_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
agent2 = Agent(state_count, action_count, head_count,
               name="2")  # "Agent2_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
agent3 = Agent(state_count, action_count, head_count,
               name="3")  # "Agent2_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
'''if not train:
    print("loading agents")
    agent1.brain.total_model.load_weights(
        "models/Environment1_500episodes.h5")
    agent2.brain.total_model.load_weights(
        "models/Environment2_500episodes.h5")'''

# ------------------------init Selector ------------------

selectorspecific = Selector([agent1, agent2])
selectorsingle = Selector([agent3])


# ---------------------------run-----------------------------------
def train_test(train_episodes, test_episodes, selectorspecific, selectorsingle, environments, name, render=False):
    # train
    train_test_logger = Logger(name=name, filename="test_train_log_{}".format(name))
    train_test_logger.add_data(["name", "entry_number", "selected or train", "reward", "standard deviation","correctly_selected(opt)"])
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
                print(entrynr)
        # test multi agents:
        selectorspecific.train = False
        for i in range(test_episodes):
            env_num, my_environment = random.choice(list(enumerate(environments)))
            selected, reward, std = my_environment.run(selectorspecific, train=False, verbose=False, render=render)
            train_test_logger.add_data(
                [str(name), str(entrynr), "test:specific:{}/{}".format(selected-1, env_num), str(reward), str(std),
                 str((selected-1) == env_num)])
            entrynr += 1
            print(entrynr)
        # single agent
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

    finally:
        for i, agent in enumerate(selectorspecific.agents):
            timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
            agent.brain.total_model.save("models/{}Trained:{}_onEnv:{}_time:{}.h5".format(name,train_episodes,i,timestamp))
        for i, agent in enumerate(selectorsingle.agents):
            timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
            agent.brain.total_model.save("models/{}Trained:{}_onAllEnv_time:{}.h5".format(name,train_episodes,timestamp))
        print("Done")


train_test(train_episodes, test_episodes, selectorspecific, selectorsingle, environments, name)
