import datetime
import random
import time
import sys
import argparse
import gym
import os.path

import lunar_lander as ll
from Environment import Environment
from DDQN_Agent import Agent
from Uncertainty_Selector import Selector
from Logger import Logger

############################################################ Main
# parser
parser = argparse.ArgumentParser(description='Process input for training models.')
parser.add_argument("--name")
# parser.add_argument("-e")
# parser.add_argument("--trainingsteps")
# parser.add_argument("--testinginterval")
# parser.add_argument("-shared_exp")
# parser.add_argument("-e")
args = parser.parse_args()
# TODO use arguments


# ------------------------define modes----------------------
verbose = True
render = True
train = False
head_count = 10
name = args.name

# ---------------define Environments-------------------------

env1 = Environment(gym.make("procgen:procgen-chaser-v0", distribution_mode="easy"), name="procgen:procgen-chaser-v0")
env2 = Environment(gym.make("procgen:procgen-bigfish-v0", distribution_mode="easy"), name="procgen:procgen-bigfish-v0")
env3 = Environment(gym.make("procgen:procgen-caveflyer-v0", distribution_mode="easy"),
                   name="procgen:procgen-caveflyer-v0")
environments = [env1, env2, env3]
# ------------------------init Agents ---------------------

input_dims = env1.environment.observation_space.shape

action_count = env1.environment.action_space.n
shared_exp = 1
steps = 2e6
if int(name) <= 6:
    shared_exp = 0.5
    steps = 2e6
elif int(name) <= 12:
    shared_exp = 0.9
    steps = 2e6
else:# int(name) <= 18:
    shared_exp = 0.9
    steps = 4e6

steps = 5e5

agent1 = Agent(input_dims, action_count, head_count, shared_exp=shared_exp, name="{}_chaser".format(name), save_memory_multiplier=0.1)
agent2 = Agent(input_dims, action_count, head_count, shared_exp=shared_exp, name="{}_bigfish".format(name), save_memory_multiplier=0.1)
agent3 = Agent(input_dims, action_count, head_count, shared_exp=shared_exp, name="{}_caveflyer".format(name), save_memory_multiplier=0.1)

agent_multi = Agent(input_dims, action_count, head_count, shared_exp=shared_exp, name="{}_multi".format(name), save_memory_multiplier=0.1)

# ------------------------init Selectors ------------------
multi_logger = Logger(filename="{}_multi_selector".format(name))
selector_multi_agent = Selector([agent1, agent2, agent3])#, logger=multi_logger)
single_logger = Logger(filename="{}_single_selector".format(name))
selector_single_agent = Selector([agent_multi])#, logger=single_logger)

# load existing agents into memory if not new
try:
    for agent in selector_multi_agent.agents:
        if os.path.isfile("models/{}.h5".format(agent.name)):
            agent.load_model()
    for agent in selector_single_agent.agents:
        if os.path.isfile("models/{}.h5".format(agent.name)):
            agent.load_model()
except:
    print("exception loading agents")


# ---------------------------run-----------------------------------


def train_and_eval_multi(training_steps, test_interval, test_episodes, selector, environments, render=True,verbose=False, name=str(name), shared_exp = shared_exp):
    multi_logger = Logger(name=name, filename="{}_multi".format(name), mode="save_on_command")
    # if it is a new training and evaluation, write parameters and headers to file
    entrynr = 0
    start = 0
    #if this is a new start
    if not os.path.isfile("logs/{}_teststate.csv".format(name)):
        # one line for all the environments and their numbers
        parameter_string = "steps: {}, shared_exp: {}".format(training_steps, shared_exp)
        multi_logger.add_data(parameter_string)
        env_list = []
        for enr, e in enumerate(environments):
            env_list.append((enr, e.name))
        multi_logger.add_data(env_list)
        # one line with parameters
        parameter_header = ["training_steps", "test_interval", "shared_experience", "head_count"]
        parameter_list = [str(training_steps), str(test_interval), str(selector.selected_agent.shared_exp),
                          str(selector.selected_agent.head_count)]
        multi_logger.add_data(parameter_header)
        multi_logger.add_data(parameter_list)
        # header with the column names after that
        header_train = ["entrynr", "train", "agentnr", "envnr", "agent_steps", "std_dev_punished", "std_dev", "reward",
                        "timestamp"]

        header_eval = ["entrynr", "eval", "envnr"]
        for i, agent in enumerate(selector.agents):
            header_eval.append("Agent{}_std_punished".format(str(i)))
            header_eval.append("Agent{}_std".format(str(i)))
        header_eval.append("winner")
        header_eval.append("reward")
        header_eval.append("correctly_selected")

        header_log = ["entrynr", "log", "entry", "timestamp"]
        multi_logger.add_data(header_train)
        multi_logger.add_data(header_eval)
        multi_logger.add_data(header_log)
        multi_logger.save_data()
        #entrynr = 1
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";"+str(start) +";" \
                        + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()
    #if there has been a start already
    else:
        #read in entrynr and i
        mylogfile = open("logs/{}_teststate.csv".format(name), 'r')
        firstRow = mylogfile.readline()
        fieldnames = firstRow.strip('\n').split(";")
        entrynr = int(fieldnames[0])
        start = int(fieldnames[1])

        #todo use new logfile mechanism of saving data only on command
    # training + testing
    for i in range(start, int(training_steps / test_interval)):
        # training
        if verbose:
            print("starting training at {} steps".format(i*test_interval))
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";" + str(i) + ";" \
                      + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()

        for nr, agent in enumerate(selector.agents):
            if verbose:
                print("training on agent nr {} until {} steps".format(nr, (i + 1) * test_interval))
            while agent.steps <= (i + 1) * test_interval:
                selector.training(nr)
                selected, reward, stds, _ = environments[nr].run(selector, train=True, verbose=False, render=render)
                if isinstance(stds, tuple):
                    pstd, std = stds
                else:
                    pstd = stds
                    std = stds
                    # ["entrynr", "train", "agentnr", "envnr", "agent_steps", "std_dev_punished", "std_dev", "reward"
                multi_logger.add_data(
                    [str(entrynr), "train", str(nr), str(nr), str(agent.steps), str(pstd), str(std), str(reward)])
                entrynr += 1

            # save agent
            agent.save_agent()
            multi_logger.add_data([str(entrynr),"log", "agent_nr_{}_saved_at_{}steps".format(nr, agent.steps)])
            entrynr += 1
            #write data from logger to disk
            multi_logger.save_data()
            #write entrynumber and i to file
            mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
            string_data = str(entrynr) + ";" + str(i) + ";" \
                          + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
            mylogfile.write(string_data)
            mylogfile.close()


        #log the end of this iteration of testing
        multi_logger.add_data([str(entrynr), "log",
                               "trained all agents for {} steps, starting evaluation".format((i+1)*test_interval)])
        entrynr += 1
        multi_logger.save_data()
        #save i and entrynr
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";" + str(i) + ";" \
                      + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()
        # testing
        selector.testing()
        if verbose:
            print("starting evaluation at {} steps".format((i + 1) * test_interval))
        # pick an environment at random for test_epsiodes, log the std of all agents and the following decision
        for test_ep in range(test_episodes):
            #test each environment for the approx. same amount
            env_nr = test_ep % len(environments)
            # getting all the standard deviations as well as the picked one in selector logger
            selected, reward, std, log = environments[env_nr].run(selector, train=False, verbose=False, render=render)
            correct = 1
            if log[-1] != env_nr:
                correct = 0
            # entrynr; eval; envnr; Agent0_std_punished; Agent0_std; Agent1_std_punished; Agent1_std; Agent2_std_punished; Agent2_std; winner; reward; correctly_selected;
            log_data = [entrynr, "eval", env_nr, log, reward, correct ]
            multi_logger.add_data(log_data)
            entrynr += 1
        multi_logger.add_data([str(entrynr),"log", "evaluation round done"])
        multi_logger.save_data()
        entrynr += 1
        if verbose:
            print("ended evaluation")

        #write state to logfile
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";" + str(start) + ";" \
                      + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()

    # finishing up training and testing:
    multi_logger.add_data([str(entrynr),"log", "all done, finished simulation"])
    if verbose:
        print("finished train and eval")
    entrynr += 1
    multi_logger.save_data()

#todo copy and adapt multi eval to eval single
def train_and_eval_single(training_steps, test_interval, test_episodes, selector, environments, render=True, name=name):
    single_logger = Logger(name=name, filename="{}_single".format(name))
    # if it is a new training and evaluation, write parameters and headers to file
    entrynr = 0
    if not os.path.isfile(single_logger.filename):
        # one line for all the environments and their numbers
        env_list = []
        for enr, e in enumerate(environments):
            env_list.append((enr, e.name))
        single_logger.add_data(env_list)
        # one line with parameters
        parameter_header = ["training_steps", "test_interval", "shared_experience", "head_count"]
        parameter_list = [str(training_steps), str(test_interval), str(selector.selected_agent.shared_exp),
                          str(selector.selected_agent.head_count)]
        single_logger.add_data(parameter_header)
        single_logger.add_data(parameter_list)
        # header with the column names after that
        header = ["entrynr", "train/eval", "agentnr", "envnr", "agent_steps", "reward", "std", "timestamp"]
        single_logger.add_data(header)
        entrynr = 1

    # training + testing
    # to correct for multiple environments
    training_steps = training_steps * len(environments)
    agent = selector.selected_agent
    for i in range(int(training_steps / test_interval)):
        # training
        selector.training(0)
        while agent.steps <= (i + 1) * test_interval:
            # randomly select an environment to "play"
            nr = random.randint(0, len(environments) - 1)
            selected, reward, std = environments[nr].run(selector, train=True, verbose=False, render=render)
            single_logger.add_data(
                [str(entrynr), "train", str(0), str(nr), str(agent.steps), str(reward), str(std)])
            entrynr += 1
        # save agent
        agent.save_agent()
        single_logger.add_data(["-2", "agent_saved"])

        # testing
        single_logger.add_data(["-1", "saved agent, starting evaluation"])
        selector.testing()
        # pick an environment at random for test_epsiodes, log the std of all agents and the following decision
        for test_ep in range(test_episodes):
            test_env_index = random.randint(0, len(environments) - 1)
            test_env = environments[test_env_index]
            # getting all the standard deviations as well as the picked one in selector logger
            selected, reward, std = test_env.run(selector, train=False, verbose=False, render=render)
            # ["entrynr", "train/eval", "agentnr", "envnr", "agent_steps", "reward", "std", "timestamp"]
            single_logger.add_data([str(entrynr), "eval", str(selected), str(test_env_index),
                                    str(selector.selected_agent.steps), str(reward), str(std)])
            entrynr += 1
        single_logger.add_data(["-3", "evaluation round done"])
    # finishing up training and testing:
    single_logger.add_data(["-3", "done"])


train_and_eval_multi(steps, int(steps/20), 50, selector_multi_agent, environments, render=render, verbose=True)

#if int(name) % 2 == 0:
    #train_and_eval_multi(steps, int(steps/20), 50, selector_multi_agent, environments, render=False)
#else:
#    train_and_eval_single(steps, int(steps/20), 50, selector_single_agent, environments, render=False)
'''
def train_test(train_episodes, test_episodes, selectorspecific, selectorsingle, environments, name, render):
    # train
    train_test_logger = Logger(name=name, filename="test_train_log_chaser_{}".format(name))
    train_test_logger.add_data(
        ["name", "entry_number", "selected or train", "reward", "standard deviation", "correctly_selected(opt)"])
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
                    [str(name), str(entrynr), "train: {}".format(selected), str(reward),
                     str(std)])  # TODO logging anpassen
                entrynr += 1
                if entrynr % 100 == 0:
                    # savemodel(agent, name)
                    model_logger.add_data(["episodes:", str(
                        entrynr)])  # TODO include in test episodes, find a way to decouple training the models and evaluating them
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
        selectorsingle.train = True
        for i in range(train_episodes):
            my_environment = random.choice(environments)
            _, reward, std = my_environment.run(selectorsingle, train=True, verbose=False, render=render)
            train_test_logger.add_data([str(name), str(entrynr), "train:generic", str(reward), str(std)])
            entrynr += 1
        # test it
        selectorsingle.train = False
        for i in range(test_episodes):
            my_environment = random.choice(environments)
            _, reward, std = my_environment.run(selectorsingle, train=False, verbose=False, render=render)
            train_test_logger.add_data(
                [str(name), str(entrynr), "test:generic{}".format(train_episodes), str(reward), str(std)])
            entrynr += 1
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
    finally:
        timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
        for i, agent in enumerate(selectorspecific.agents):
            agent.online_network.total_model.save(
                "models/{}_{}_episodes:{}_{}.h5".format(name, agent.name, train_episodes,
                                                        timestamp))  # {}Trained:{}_onEnv:{}_time:{}.h5".format(name,train_episodes,i,timestamp))
        for i, agent in enumerate(selectorsingle.agents):
            agent.online_network.total_model.save(
                "models/{}_{}_episodes:{}_{}_.h5".format(name, agent.name, train_episodes, timestamp))
        print("Done")
'''
# train_test(train_episodes, test_episodes, selector_multi_agent, selector_multi_agent, environments, name, render=render)
