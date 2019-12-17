import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def read_in(data):
    # data = ["data1.csv", "data2.csv"]
    i = 0
    dataframes = []
    for d in data:
        df = pd.read_csv(d, header=None, index_col=None, sep=';')
        header = []
        df['label'] = i
        i += 1
        dataframes.append(df)
    dataset = pd.concat(dataframes)
    return dataset


class Selector:
    # data is a list of names of datafiles in the same order as the agents
    # agents is a list of agents that correspond to the data
    def __init__(self, agents, data):
        self.classifier = KNeighborsClassifier(n_neighbors=5)
        self.dataset = read_in(data)
        self.agents = agents
        self.selected_agent = agents[0]
        self.reclassify()

    # reads in data from csv, adds their index as a last row and then

    def reclassify(self):
        x = self.dataset.iloc[:, :-1]
        y = self.dataset.iloc[:, self.dataset.shape[1] - 1]
        self.classifier.fit(x, y)

    def act(self, state):
        return self.selected_agent.act(state)

    def observe(self, sample):
        self.selected_agent.observe(sample)

    def replay(self):
        self.selected_agent.replay()

    def select(self, state):
        c = self.classifier.predict([state])
        self.selected_agent = self.agents[c[0]]
