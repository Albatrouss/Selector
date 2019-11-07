from sklearn.neighbors import KNeighborsClassifier
import pandas



class Selector:
    def __init__(self, datafiles):
        self.datafiles = datafiles
        dataset = pandas.read_csv(datafiles, header=None)
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 11].values
        self.classifier = KNeighborsClassifier(n_neighbors=5)
        self.classifier.fit(x, y)

    def predictOne(self, state):
        return self.classifier.predict([state])
    def predictMany(self, states):
        return self.classifier.predict(states)
