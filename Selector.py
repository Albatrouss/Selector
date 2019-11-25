import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class Selector:
    # data is a list of names of datafiles in the same order as the agents
    # agents is a list of agents that correspond to the data
    def __init__(self, agents, data):
        self.dataset = self.readin(data)
        self.agents = agents
        self.selectedagent  = agents[0]
        self.reclassify()

    #reads in data from csv, adds their index as a last row and then
    def readin(self, data):
        #data = ["data1.csv", "data2.csv"]
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

    def reclassify(self):
        x = self.dataset.iloc[:,:-1]
        y = self.dataset.iloc[:,self.dataset.shape[1]-1]
        self.classifier = KNeighborsClassifier(n_neighbors=5)
        self.classifier.fit(x, y)

    def act(self, state):
        return self.selectedagent.act(state)

    def observe(self, sample):
        self.selectedagent.observe(sample)

    def replay(self):
        self.selectedagent.replay()

    def select(self, state):
        c = self.classifier.predict([state])
        self.selectedagent = self.agents[c[0]]
