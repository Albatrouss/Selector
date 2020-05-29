import datetime
import os
import random
import time
import sys

import lunar_lander as ll
from Environment import Environment
from DDQN_Agent import Agent
from Uncertainty_Selector import Selector
from Logger import Logger

############################################################ Main
# ------------------------define modes----------------------
verbose = True
render = True
train = False

head_count = 10
name = sys.argv[1]

# ---------------define Environments-------------------------
my_problem1 = ll.LunarLanderModable()
my_problem1.seed(int(time.time()))
my_problem2 = ll.LunarLanderModable()
my_problem2.seed(int(time.time()))
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

state_count = my_environment1.environment.observation_space.shape
action_count = my_environment1.environment.action_space.n
'''
self, input_dim, action_count, head_count, name, steps_before_learning=10000, epsilon=1,
      epsilon_decay=0.00000099, epsilon_min=0.01, batch_size=32, target_update_intervall=10000,
      memory_capacity=1000000, alpha=0.00025, shared_exp=0.8, gamma=0.99, mode="conv",
      save_interval=1000000, save_big_path="/big/h/haemmerle/", save_memory_multiplier = 0.1):

'''
shared_exp = 0.8

steps = 1e6
epsilon_decay = 2 / steps
#single = False

if int(name) <= 10:
    epsilon_max = 0
    epsilon_min = 0
'''
if int(name) <= 5:
    epsilon_max = 0
    epsilon_min = 0
    single = False
elif int(name) <= 10:
    epsilon_max = 0
    epsilon_min = 0
    single = True
elif int(name) <= 15:
    epsilon_max = 1
    epsilon_min = 0.1
    single = False
elif int(name) <= 20:
    epsilon_max = 1
    epsilon_min = 0.1
    single = True
    '''

agent1 = Agent(state_count, action_count, head_count,
               name="{}_l".format(name), steps_before_learning=10000, epsilon_decay=epsilon_decay,
               memory_capacity=100000, shared_exp=shared_exp, mode="simple", save_interval=1e5,
               # save_big_path="models/",
               save_memory_multiplier=0.1, epsilon=epsilon_max, epsilon_min=epsilon_min)
agent2 = Agent(state_count, action_count, head_count,
               name="{}_r".format(name), steps_before_learning=10000, epsilon_decay=epsilon_decay,
               memory_capacity=100000, shared_exp=shared_exp, mode="simple", save_interval=1e5,
               # save_big_path="models/",
               save_memory_multiplier=0.1, epsilon=epsilon_max, epsilon_min=epsilon_min)

# agent2 = Agent(state_count, action_count, head_count,
#               name="{}_r".format(name))  # "Agent2_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
agent3 = Agent(state_count, action_count, head_count,
               name="{}_s".format(name), steps_before_learning=10000, epsilon_decay=epsilon_decay,
               memory_capacity=100000, shared_exp=shared_exp, mode="simple", save_interval=1e5,
               # save_big_path="models/",
               save_memory_multiplier=0.1, epsilon=epsilon_max, epsilon_min=epsilon_min)
#              name="3")  # "Agent2_hc{}_train{}_episodes{}".format(head_count, train, train_episodes))
'''if not train:
    print("loading agents")
    agent1.brain.total_model.load_weights(
        "models/Environment1_500episodes.h5")
    agent2.brain.total_model.load_weights(
        "models/Environment2_500episodes.h5")'''

# ------------------------init Selectors ------------------

selector_specific = Selector([agent1, agent2])
selector_single = Selector([agent3])


# ---------------------------run-----------------------------------
''' no longer used
def train_test(train_episodes, test_episodes, selectorspecific, selectorsingle, environments, name, render=False):
    # train
    train_test_logger = Logger(name=name, filename="test_train_log_{}".format(name))
    train_test_logger.add_data(
        ["name", "entry_number", "selected or train", "reward", "standard deviation", "correctly_selected(opt)"])
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
                [str(name), str(entrynr), "test:specific:{}/{}".format(selected, env_num), str(reward), str(std),
                 str(selected == env_num)])
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
            agent.brain.total_model.save(
                "models/{}Trained:{}_onEnv:{}_time:{}.h5".format(name, train_episodes, i, timestamp))
        for i, agent in enumerate(selectorsingle.agents):
            timestamp = str(datetime.datetime.fromtimestamp(time.time()).isoformat())
            agent.brain.total_model.save(
                "models/{}Trained:{}_onAllEnv_time:{}.h5".format(name, train_episodes, timestamp))
        print("Done")
'''

def train_and_eval(training_steps, test_interval, test_episodes, selector, environments, render=False,
                         verbose=False, name="LL" + str(name), shared_exp=shared_exp, epsilon_max=epsilon_max,
                         single=single):
    multi_logger = Logger(name=name, filename="{}_multi".format(name), mode="save_on_command")
    # if it is a new training and evaluation, write parameters and headers to file
    entrynr = 0
    start = 0
    # if this is a new start
    if not os.path.isfile("logs/{}_teststate.csv".format(name)):
        # one line for all the environments and their numbers
        parameter_string = "steps: {}, shared_exp: {}".format(training_steps, shared_exp)
        multi_logger.add_data(parameter_string)
        env_list = []
        for enr, e in enumerate(environments):
            env_list.append((enr, e.name))
        multi_logger.add_data(env_list)
        # one line with parameters
        parameter_header = ["training_steps", "test_interval", "shared_experience", "head_count", "epsilon-greedy",
                            "single"]
        parameter_list = [str(training_steps), str(test_interval), str(selector.selected_agent.shared_exp),
                          str(selector.selected_agent.head_count), str(epsilon_max), str(single)]
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
        # entrynr = 1
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";" + str(start) + ";" \
                      + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()
    # if there has been a start already
    else:
        # read in entrynr and i
        mylogfile = open("logs/{}_teststate.csv".format(name), 'r')
        firstRow = mylogfile.readline()
        fieldnames = firstRow.strip('\n').split(";")
        entrynr = int(fieldnames[0])
        start = int(fieldnames[1])
        # todo use new logfile mechanism of saving data only on command
    # training + testing
    for i in range(start, int(training_steps / test_interval)):
        # training
        if verbose:
            print("starting training at {} steps".format(i * test_interval))
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
                selected, reward, stds, _ = environments[nr].run(selector, train=True, verbose=verbose, render=render)
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
            multi_logger.add_data([str(entrynr), "log", "agent_nr_{}_saved_at_{}steps".format(nr, agent.steps)])
            entrynr += 1
            # write data from logger to disk
            multi_logger.save_data()
            # write entrynumber and i to file
            mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
            string_data = str(entrynr) + ";" + str(i) + ";" \
                          + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
            mylogfile.write(string_data)
            mylogfile.close()

        # log the end of this iteration of testing
        multi_logger.add_data([str(entrynr), "log",
                               "trained all agents for {} steps, starting evaluation".format((i + 1) * test_interval)])
        entrynr += 1
        multi_logger.save_data()
        # save i and entrynr
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
            # test each environment for the approx. same amount
            env_nr = test_ep % len(environments)
            # getting all the standard deviations as well as the picked one in selector logger
            selected, reward, std, log = environments[env_nr].run(selector, train=False, verbose=verbose, render=render)
            correct = 1
            if log[-1] != env_nr:
                correct = 0
            # entrynr; eval; envnr; Agent0_std_punished; Agent0_std; Agent1_std_punished; Agent1_std; Agent2_std_punished; Agent2_std; winner; reward; correctly_selected;
            log_data = [entrynr, "eval", env_nr, log, reward, correct]
            multi_logger.add_data(log_data)
            entrynr += 1
        multi_logger.add_data([str(entrynr), "log", "evaluation round done"])
        multi_logger.save_data()
        entrynr += 1
        if verbose:
            print("ended evaluation")

        # write state to logfile
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";" + str(i) + ";" \
                      + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()

    # finishing up training and testing:
    mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
    string_data = str(entrynr) + ";" + str(sys.maxsize) + ";" \
                  + "done;" + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
    mylogfile.write(string_data)

    multi_logger.add_data([str(entrynr), "log", "all done, finished simulation"])
    if verbose:
        print("finished train and eval")
    entrynr += 1
    multi_logger.save_data()


def train_and_eval_single(training_steps, test_interval, test_episodes, selector, environments, render=False,
                          verbose=False, name="LL" + str(name), shared_exp=shared_exp, epsilon_max=epsilon_max,
                          single=single):
    multi_logger = Logger(name=name, filename="{}_single".format(name), mode="save_on_command")
    # if it is a new training and evaluation, write parameters and headers to file
    entrynr = 0
    start = 0
    # if this is a new start
    if not os.path.isfile("logs/{}_teststate.csv".format(name)):
        # one line for all the environments and their numbers
        parameter_string = "steps: {}, shared_exp: {},".format(training_steps, shared_exp)
        multi_logger.add_data(parameter_string)
        env_list = []
        for enr, e in enumerate(environments):
            env_list.append((enr, e.name))
        multi_logger.add_data(env_list)
        # one line with parameters
        parameter_header = ["training_steps", "test_interval", "shared_experience", "head_count", "epsilon-greedy"]
        parameter_list = [str(training_steps), str(test_interval), str(selector.selected_agent.shared_exp),
                          str(selector.selected_agent.head_count), str(epsilon_max)]
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
        # entrynr = 1
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";" + str(start) + ";" \
                      + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()
    # if there has been a start already
    else:
        # read in entrynr and i
        mylogfile = open("logs/{}_teststate.csv".format(name), 'r')
        firstRow = mylogfile.readline()
        fieldnames = firstRow.strip('\n').split(";")
        entrynr = int(fieldnames[0])
        start = int(fieldnames[1])
        # todo use new logfile mechanism of saving data only on command
    # training + testing
    for i in range(start, int(training_steps / test_interval)):
        # training
        if verbose:
            print("starting training at {} steps".format(i * test_interval))
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
                selected, reward, stds, _ = environments[nr].run(selector, train=True, verbose=verbose, render=render)
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
            multi_logger.add_data([str(entrynr), "log", "agent_nr_{}_saved_at_{}steps".format(nr, agent.steps)])
            entrynr += 1
            # write data from logger to disk
            multi_logger.save_data()
            # write entrynumber and i to file
            mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
            string_data = str(entrynr) + ";" + str(i) + ";" \
                          + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
            mylogfile.write(string_data)
            mylogfile.close()

        # log the end of this iteration of testing
        multi_logger.add_data([str(entrynr), "log",
                               "trained all agents for {} steps, starting evaluation".format((i + 1) * test_interval)])
        entrynr += 1
        multi_logger.save_data()
        # save i and entrynr
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
            # test each environment for the approx. same amount
            env_nr = test_ep % len(environments)
            # getting all the standard deviations as well as the picked one in selector logger
            selected, reward, std, log = environments[env_nr].run(selector, train=False, verbose=verbose, render=render)
            correct = 1
            if log[-1] != env_nr:
                correct = 0
            # entrynr; eval; envnr; Agent0_std_punished; Agent0_std; Agent1_std_punished; Agent1_std; Agent2_std_punished; Agent2_std; winner; reward; correctly_selected;
            log_data = [entrynr, "eval", env_nr, log, reward, correct]
            multi_logger.add_data(log_data)
            entrynr += 1
        multi_logger.add_data([str(entrynr), "log", "evaluation round done"])
        multi_logger.save_data()
        entrynr += 1
        if verbose:
            print("ended evaluation")

        # write state to logfile
        mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
        string_data = str(entrynr) + ";" + str(i) + ";" \
                      + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        mylogfile.close()

    # finishing up training and testing:
    mylogfile = open("logs/{}_teststate.csv".format(name), 'w+')
    string_data = str(entrynr) + ";" + str(sys.maxsize) + ";" \
                  + "done;" + str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
    mylogfile.write(string_data)

    multi_logger.add_data([str(entrynr), "log", "all done, finished simulation"])
    if verbose:
        print("finished train and eval")
    entrynr += 1
    multi_logger.save_data()


train_and_eval(steps, 1e5, 20, selector_specific, environments, render=False, verbose=True)

# train_test(train_episodes, test_episodes, selector_specific, selector_single, environments, name)
