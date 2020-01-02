import numpy

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

    def act(self, state, mode='preselect'):
        if mode == 'preselect':
            action = self.selected_agent.act_on_head(state=state, head_num=self.selected_head_num)
            return action
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

        if self.train or mode == 'train':
            taction, tstd = self.selected_agent.act(state)
            return taction

    def select(self, state):
        if self.train:
            #select a head to use for the episode
            #todo remove, only for getting a standard deviation purpose
            _, std = self.selected_agent.act(state)
            #end todo
            self.selected_head_num = numpy.random.choice(range(self.selected_agent.head_count))
            return self.selected_agent.name, self.selected_head_num, std
        action_std = []
        for i in range(len(self.agents)):
            agent = self.agents[i]
            action, std = agent.act(state)
            action_std.append((i, action, std))

        # choose agent with lowest standard deviation
        #print(action_std)
        agent_num, best_action, std = min(action_std, key=lambda k: k[2])
        self.selected_agent = self.agents[agent_num]
        # select a head to follow for this episode
        self.selected_head_num = numpy.random.choice(range(self.selected_agent.head_count))
        return agent_num, self.selected_head_num, std

    def observe(self, sample):
        self.selected_agent.observe(sample)

    def replay(self):
        self.selected_agent.replay()
