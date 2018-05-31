import sys

class ProcessWriter(object):
    """Class to handle writing from parallel processes
    
    Attributes:
        process (int): the process number
    
    """
    
    def __init__(self, process = 0):
        self.process = process
    
    def writeString(self, text):
        pre = ("Process %d: " % self.process)
        return pre + str(text).replace('\n', '\n' + pre)
    
    def write(self, text):        
        print(self.writeString(text))
        sys.stdout.flush()

    