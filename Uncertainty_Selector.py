import numpy
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier


class Selector:
    # agents is a list of agents that correspond to the data
    def __init__(self, agents):
        self.agents = agents
        self.selected_agent = agents[0]
        self.selected_head_num = 0
        self.train = False

    def training(self, agent_num):
        # overrides the selectionprocess to train the agent
        self.selected_agent = self.agents[agent_num]
        self.train = True

    '''
    we have different modes:
        mode :  'train': only return on selected agent, use normal act
                'continuous': select for every action the most sure agent, do not use normal act, only act_on_model
                'explicit': only select a new agent when the function select is called
    '''

    def act(self, state, mode='explicit'):
        if mode == 'explicit':
            action, std = self.selected_agent.act(state)
            return action
        if mode == 'continuous':
            action_std = []
            for i in range(len(self.agents)):
                agent = agents[i]
                action, std = agent.act(state)
                action_std.append((i, action, std))

            # choose agent with lowest standard deviation TODO rework with seleciton criteria
            agent_num, best_action, std = min(action_std, key=lambda k: k[2])
            self.selected_agent = self.agents[agent_num]
            # select a head to follow
            self.selected_head_num = numpy.random.choice(range(self.selected_agent.headcount))

        if self.train or mode == 'train':
            taction, tstd = self.selected_agent.act(state)
            return taction

    def select(self, state):
        if self.train:
            return None
        action_std = []
        for i in range(len(self.agents)):
            agent = self.agents[i]
            action, std = agent.act(state)
            action_std.append((i, action, std))

        # choose agent with lowerst standard deviation
        print(action_std)
        agent_num, best_action, std = min(action_std, key=lambda k: k[2])
        self.selected_agent = self.agents[agent_num]
        # select a head to follow for this episode
        self.selected_head_num = numpy.random.choice(range(self.selected_agent.headcount))
        return agent_num, self.selected_head_num, std

    def observe(self, sample):
        self.selected_agent.observe(sample)

    def replay(self):
        self.selected_agent.replay()
