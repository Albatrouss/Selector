import random
import lunar_lander as ll
from Environment import Environment
from Agent import Agent
from Selector import Selector

############################################################ Main
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
train = True

state_count1 = my_environment1.environment.observation_space.shape[0]
action_count1 = my_environment1.environment.action_space.n

state_count2 = my_environment2.environment.observation_space.shape[0]
action_count2 = my_environment2.environment.action_space.n

agent1 = Agent(state_count1, action_count1)
agent2 = Agent(state_count2, action_count2)
if not train:
    agent2.loadModel("models/ll2.h5", 1)
    agent1.loadModel("models/ll1.h5", 1)

# ------------------------init Selector ------------------

selector = Selector([agent1, agent2], ["data/data1.csv",
                                       "data/data2.csv"])

# ---------------------------run-----------------------------------

i = 0
verbose = True

try:
    while i < 500:
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
        agent1.brain.model.save("models/ll1.2.h5")
        agent2.brain.model.save("models/ll2.2.h5")
    print("Done")
