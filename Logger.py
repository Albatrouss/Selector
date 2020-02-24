import datetime
import time
from concurrent.futures import ThreadPoolExecutor


class Logger:
    def __init__(self, name="", filename=None, mode="save_always"):
        self.executor = ThreadPoolExecutor(max_workers=1)
        if mode == "save_always":
            self.save_on_command = False
        elif mode == "save_on_command":
            self.save_on_command = True
            self.towrite = []
        if filename is None:
            self.filename = "logs/{}{}.csv".format(name, datetime.datetime.fromtimestamp(time.time()).isoformat())
        else:
            self.filename = "logs/{}.csv".format(filename)
        # print(self.filename)

    def add_data(self, data):
        if self.save_on_command: # save on command
            self.towrite.append(self.extract(data))
        else: #save always
            self.executor.submit(self.add_data_, data)

    def add_data_(self, data):
        # add data including the timestamp to the log file (csv)
        string_data = self.extract(data)

        mylogfile = open(self.filename, 'a+')
        mylogfile.write(string_data)
        mylogfile.close()

    #def save_data(self):
    #    self.executor.submit(self.save_data)

    def save_data(self):
        if self.save_on_command:
            mylogfile = open(self.filename, 'a+')
            for entry in self.towrite:
                mylogfile.write(entry)
            mylogfile.close()
            self.towrite = []

    def extract(self, data):
        extraction = []

        self.extract_(data, extraction)

        string_data = ""
        for d in extraction:
            string_data += (str(d) + ";")

        string_data += str(datetime.datetime.fromtimestamp(time.time()).isoformat()) + "\n"
        return string_data

    def extract_(self, data, extraction):

        if not isinstance(data, (list, tuple)):
            extraction.append(data)
        elif data is not None:
            for d in data:
                self.extract_(d, extraction)
