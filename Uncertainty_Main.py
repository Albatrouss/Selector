import datetime
import random
import time

import lunar_lander as ll
from Environment import Environment
from Uncertainty_Agent import Agent
from Uncertainty_Selector import Selector
from Logger import Logger

############################################################ Main
# ------------------------define modes----------------------
verbose = True
render = True
train = True
train_episodes = 500
test_episodes = 200
head_count = 10

# ---------------------define Loggers ------------------------------------------------------------
main_logger = Logger(name="Main_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
main_logger.add_data(["Episode_num; Environment_num;Agent_num;Head_num; Reward; Timestamp"])
train_logger1 = Logger(name="training1", filename="1_outcome.csv")
train_logger2 = Logger(name="training2", filename="2_outcome.csv")
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

# ------------------------init Agents ---------------------

state_count1 = my_environment1.environment.observation_space.shape[0]
action_count1 = my_environment1.environment.action_space.n

state_count2 = my_environment2.environment.observation_space.shape[0]
action_count2 = my_environment2.environment.action_space.n

agent1 = Agent(state_count1, action_count1, head_count,
               name="1")  # "Agent1_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
agent2 = Agent(state_count2, action_count2, head_count,
               name="2")  # "Agent2_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
'''
if not train:
    print("loading agents")
    agent1.brain.total_model.load_weights(
        "models/Trueuncert1_hc10_trainepisodes200_time2019-12-12T17:23:30.800531.h5")
    agent2.brain.total_model.load_weights(
        "models/Trueuncert2_hc10_trainepisodes200_time2019-12-12T17:23:32.381833.h5")'''

# ------------------------init Selector ------------------

selector = Selector([agent1, agent2])

# ---------------------------run-----------------------------------
finished = False
i = 0
try:
    # train agent 1 if train == true
    selector.training(0)
    my_environment = my_environment1
    while i < train_episodes and train:
        selected, reward = my_environment.run(selector, train, verbose, render=render)
        train_logger1.add_data([i, selected, reward])
        i += 1
        print("Agent 1: Episode {}/{}".format(i, train_episodes))
    i = 0
    my_environment = my_environment2
    selector.training(1)
    # train agent 2 if train == true
    while i < train_episodes and train:
        selected, reward = my_environment.run(selector, train, verbose, render=render)
        train_logger1.add_data([i, selected, reward])
        i += 1
        print("Agent 2: Episode {}/{}".format(i, train_episodes))
    i = 0
    selector.train = False
    while i < test_episodes:
        data = [i]
        print("Episode {} started".format(i))
        env = random.randint(0, 1)
        data.append(env)
        if env == 0:
            my_environment = my_environment1
            print("Environment 0 loaded")
        if env == 1:
            my_environment = my_environment2
            print("Environment 1 loaded")

        data.append(my_environment.run(selector, train=False, verbose=verbose, render=render))
        main_logger.add_data(data)
        i = i + 1
    finished = True

finally:
    timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
    agent1.brain.total_model.save(
        "models/{}uncert1_hc{}_trainepisodes{}_time{}.h5".format(finished, head_count, train_episodes, timestamp))
    agent2.brain.total_model.save(
        "models/{}uncert2_hc{}_trainepisodes{}_time{}.h5".format(finished, head_count, train_episodes, timestamp))
    print("Savedanddone")
