
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from growthrate_2024 import GrowthRate
import csv
import glob
import shutil

global fname 
fname = 'sim2_10_07_2024.csv'

def keypress(event):
    if event.key == ' ':
        plt.close()
        gr.save(fname= fname ,header=['qpv','slope','error'])
    else:
        gr.slope=0
        gr.save(fname= fname ,header=['qpv','slope','error'])
        plt.close() 



simfiles=glob.glob('results2/*.h5')

#measdata=pd.read_parquet(meas) \n pos=measdata['SPS.BQ.KICKED/ContinuousAcquisition'][0]['value']['rawDataV']
print(simfiles)
for i,sim in enumerate(simfiles):
    qpv = float(sim.split('/')[1].split('qpv')[1].split('_')[0])
    
    gr=GrowthRate(file=sim,qpv=qpv)
    try:
        gr.MWFFT_analysis()
    except:
        gr=GrowthRate(file=sim,qpv=qpv, mode='interactive')
        gr.MWFFT_analysis()
        
    print(f'progress: {i}/{len(sim)}')
    gr.plot()
    #gr.save(fname=fname,header=['qpv','slope','error'] )

    plt.gcf().canvas.mpl_connect('key_press_event', keypress)
    
    



