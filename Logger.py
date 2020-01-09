import datetime
import time
from concurrent.futures import ThreadPoolExecutor


class Logger:
    def __init__(self, name, filename=None):
        self.executor = ThreadPoolExecutor(max_workers=1)
        if filename is None:
            self.filename = "logs/{}{}.csv".format(name, datetime.datetime.fromtimestamp(time.time()).isoformat())
        else:
            self.filename = "logs/{}.csv".format(filename)
        # print(self.filename)

    def add_data(self, data):
        self.executor.submit(self.add_data_, data)

    def add_data_(self, data):
        # add data including the timestamp to the log file (csv)
        string_data = ""
        extracted = []
        self.extract(data, extracted)
        mylogfile = open(self.filename, 'a+')
        for d in extracted:
            string_data += (str(d) + ";")
        string_data += str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        mylogfile.write(string_data)
        # print("written data to {}".format(self.filename))
        mylogfile.close()

    def extract(self, data, extraction):
        if not isinstance(data, (list, tuple)):
            extraction.append(data)
        elif data is not None:
            for d in data:
                self.extract(d, extraction)
