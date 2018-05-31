import os

IlastikPath = r'C:\Program Files\ilastik-1.2.0rc10'
ElastixPath = r'C:\Program Files\elastix_v4.8'

def setup():
    global IlastikPath, ElastixPath
    
    # check existence:
    if not ElastixPath is None:
        if not os.path.exists(ElastixPath):
            #raise RuntimeWarning('Settings: elastix path %s does not exists, cf. Settings.py or type help(Settings) for details.' % ElastixPath)
            print( 'Settings: elastix path %s does not exists, cf. Settings.py or type help(Settings) for details.' % ElastixPath )
            ElastixPath = None
    
    if not IlastikPath is None:
        if not os.path.exists(IlastikPath):
            #raise RuntimeWarning('Settings: ilastik path %s does not exists, cf. Settings.py or type help(Settings) for details.' % IlastikPath)
            print( 'Settings: ilastik path %s does not exists, cf. Settings.py or type help(Settings) for details.' % IlastikPath )
            IlastikPath = None

setup()


def clarityPath():
    fn = os.path.split(__file__)
    fn = os.path.abspath(fn[0])
    return fn

clarityPath = clarityPath()
