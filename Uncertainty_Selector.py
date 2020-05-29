import numpy


class NormingMemory:
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.factor = 1
        self.memory.append(1)

    def add(self, number):
        # negative numbers are invalid standard deviations
        if number < 0:
            return
        # pop one if full
        if len(self.memory) > self.size:
            self.memory.pop(0)
        self.memory.append(number)

    def get_factor(self):
        self.factor = numpy.mean(self.memory)
        return self.factor



class Selector:
    # agents is a list of agents that correspond to the data
    def __init__(self, agents, name="selector", logger=None, memsize=500):
        self.agents = agents
        self.memories = [NormingMemory(memsize) for i in range(len(agents))]
        self.selected_agent = agents[0]
        self.selected_memory = self.memories[0]
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
        self.selected_memory = self.memories[agent_num]
        self.train = True
        self.selected_agent.train = True

    def testing(self):
        self.train = False
        for agent in self.agents:
            agent.train = False

    def act(self, state, mode='eval'):
        if mode == 'eval':
            action, stddev = self.selected_agent.act_on_head(state=state, head_num=self.selected_head_num)
            return action, stddev
        if self.train or mode == 'train':
            taction, tstd = self.selected_agent.act(state)
            return taction

    def select(self, state):
        if self.train:
            # select a head to use for the episode stddev = (pun, std)
            _, stddevs = self.selected_agent.act(state)
            if isinstance(stddevs, tuple):
                pstd, std = stddevs
            else:
                pstd = stddevs
                std = stddevs
            self.selected_memory.add(pstd)
            self.selected_agent.steps -= 1
            self.selected_head_num = numpy.random.choice(range(self.selected_agent.head_count))
            return self.selected_agent.name, stddevs, stddevs
        else:
            action_std = []
            log = []
            for i in range(len(self.agents)):
                agent = self.agents[i]
                action, stddev = agent.act(state)
                pstd, std = stddev
                log.append(stddev)
                normfactor = self.memories[i].get_factor()
                action_std.append((i, action, pstd/normfactor))
            agent_num, best_action, std = min(action_std, key=lambda k: k[2])
            self.selected_agent = self.agents[agent_num]
            self.selected_memory = self.memories[agent_num]
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
