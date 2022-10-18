from pyjapcscout import PyJapcScout
from helper_functions import callback_core
from datetime import date
import PlottingClassesSPS as pc


def outer_callback(plot_func_list):
    def callback(data, h):
        callback_core(data, h, plot_func_list)
    return callback


# Set up monitor
selector = 'SPS.USER.MD5'
BBQ_device = 'SPS.BQ.KICKED/ContinuousAcquisition'
devices = [BBQ_device, 'SPSBEAM/QPV', 'SPSBEAM/QPH']

# start PyJapcScout and so incaify Python instance
myPyJapc = PyJapcScout(incaAcceleratorName='SPS')

myMonitor = myPyJapc.PyJapcScoutMonitor(selector, devices)

# Saving data configuration
date=str(date.today())
myMonitor.saveDataPath = f'./data/growthrate/{selector}/{date}/'
myMonitor.saveData = True
myMonitor.saveDataFormat = 'parquet'

# Start acquisition
myMonitor.startMonitor()


'''
if 0:
    ## for controlling data acquisition:
    #
    myMonitor.saveData = True
    # myMonitor.saveData = False
    
    ## to stop the monitor
    #myMonitor.stopMonitor()
'''
