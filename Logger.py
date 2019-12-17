import datetime, time


class Logger:
    def __init__(self, name):
        self.mylogfile = open("logs/{}{}.csv".format(name, datetime.datetime.fromtimestamp(time.time()).isoformat()),
                              'w+')

    def add_data(self, data):
        # add data including the timestamp to the log file (csv)
        string_data = ""
        extracted = []
        self.extract(data, extracted)
        for d in extracted:
            string_data += (str(d) + ";")
        string_data += str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        self.mylogfile.write(string_data)

    def close(self):
        self.mylogfile.close()

    def extract(self, data, extraction):
        if not isinstance(data, (list, tuple)):
            extraction.append(data)
        elif data is not None:
            for d in data:
                self.extract(d, extraction)
