import numpy


class Selector:
    # agents is a list of agents that correspond to the data
    def __init__(self, agents, name="selector", logger=None):
        self.agents = agents
        self.selected_agent = agents[0]
        self.selected_head_num = 0
        self.name = name
        self.train = False
        self.logging = False
        if logger is not None:
            self.logger = logger
            self.logging = True

    def training(self, agent_num):
        # overrides the selectionprocess to train the agent
        self.selected_agent = self.agents[agent_num]
        self.train = True
        self.selected_agent.train = True

    def testing(self):
        self.train = False
        for agent in self.agents:
            agent.train = False

    def act(self, state, mode='preselect'): #TODO in train und eval teilen, nicht in preselect und whatever else
        if mode == 'preselect':
            action, stddev = self.selected_agent.act_on_head(state=state, head_num=self.selected_head_num)
            return action, stddev
        '''
        if mode == 'continuous':
            action_std = []
            for i in range(len(self.agents)):
                agent = self.agents[i]
                action, std = agent.act(state)
                action_std.append((i, action, std))

            # choose agent with lowest standard deviation TODO rework with selection criteria
            agent_num, best_action, std = min(action_std, key=lambda k: k[2])
            self.selected_agent = self.agents[agent_num]
            # select a head to follow
            self.selected_head_num = numpy.random.choice(range(self.selected_agent.headcount))
            # TODO return action?'''
        if self.train or mode == 'train':
            taction, tstd = self.selected_agent.act(state)
            return taction

    def select(self, state):
        if self.train:
            # select a head to use for the episode
            _, stddevs = self.selected_agent.act(state)
            self.selected_agent.steps -= 1
            self.selected_head_num = numpy.random.choice(range(self.selected_agent.head_count))
            return self.selected_agent.name, stddevs , stddevs
        else:
            action_std = []
            log = []
            for i in range(len(self.agents)):
                agent = self.agents[i]
                action, stddev = agent.act(state)
                pstd, std = stddev
                log.append(stddev)
                action_std.append((i, action, pstd))
            agent_num, best_action, std = min(action_std, key=lambda k: k[2])
            self.selected_agent = self.agents[agent_num]
            log.append(agent_num)
            if self.logging:
                data = [(str(agentnr), str(std)) for agentnr, _, std in action_std]
                data.append(("winner", str(agent_num)))
                self.logger.add_data(data)

            # select a head to follow for this episode
            self.selected_head_num = numpy.random.choice(range(self.selected_agent.head_count))
            return agent_num, log, std

    def observe(self, sample):
        self.selected_agent.observe(sample)

    def replay(self):
        self.selected_agent.replay()
