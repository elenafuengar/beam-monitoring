import glob
import numpy as np
import matplotlib.pyplot as plt
import datascout as ds
import pandas as pd
import csv


def onclick(event):
    global grstart
    grstart = event.xdata
    plt.close()
 
#paths
path = '/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/growthrate/SPS.USER.MD3/2022-08-30/edelafue/'
gr_path = '/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/growthrate/SPS.USER.MD3/2022-08-30/'
i_path ='/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/bct/SPS.USER.MD3/'

#read csv
df = pd.read_csv(path + 'growth_rate_values.csv', index_col=False)

#for i in range(len(df['time'])):
for i in range(1):
    #gr parquet
    time = sorted(df['time'])[i]
    time = time.split('.')
    filter = time[0]+'.'+time[1]+'.'+time[2]+'.'+time[3]+'.'+time[4]+'.*'

    gr_parquet = sorted(glob.glob(gr_path+filter))[0]
    gr = ds.parquet_to_dict(gr_parquet)

    #gr 
    vpos=gr['SPS.BQ.KICKED/ContinuousAcquisition']['value']['rawDataV']
    revf=gr['SPS.BQ.KICKED/ContinuousAcquisition']['value']['frevFreq']
    turns=range(len(vpos))
    ms=turns/np.mean(revf)

    #acq time
    filename = gr_parquet.split('/')[-1]
    time_stamp = filename.split('.parquet')[0]

    #plot
    fig, ax = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(20,6)
    ax.plot(ms, vpos, color='b')
    ax.set_xlim(5, 1000)
    ax.set_ylim(-1.0e8, 1.0e8)
    ax.set_ylabel('Vertical position', color = 'b')
    ax.set_xlabel('Time [ms]')
    ax.tick_params(axis='y', colors='b')

    #get instability start 
    grstart = 0 
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    try:
        #intensity 
        inty_parquet = sorted(glob.glob(i_path+filter))[0]
        inty = ds.parquet_to_dict(inty_parquet)

        yinty=inty['SPS.BCTDC.51456/Acquisition']['value']['totalIntensity']
        xinty=inty['SPS.BCTDC.51456/Acquisition']['value']['measStamp']
        exp=inty['SPS.BCTDC.51456/Acquisition']['value']['totalIntensity_unitExponent']
        grinty = yinty[np.where(xinty >= grstart)[0][0]]

        #plot with intensity
        xstart = np.where(ms >= grstart)[0][0]
        xpos = np.where(ms >= grstart)[0]
        xend = np.argmax(vpos[xpos])+xpos[0]

        fig, ax = plt.subplots(1,1)
        ax.axhline(grstart,c='r')
        ax.plot(ms, vpos, color='b', alpha = 0.6)
        ax.plot(ms[xstart:xend], vpos[xstart:xend], color='b') 
        ax.axvline(grstart,c='r')
        ax.set_xlim(5, 1000)
        ax.set_ylim(-1.0e8, 1.0e8)
        ax.set_ylabel('Vertical position', color = 'b')
        ax.set_xlabel('Time [ms]')
        ax.tick_params(axis='y', colors='b')
        ax2=ax.twinx()
        ax2.plot(xinty, yinty*10**exp, color='orange', alpha = 0.5)
        ax2.set_ylabel('Intensity [p/b]', color = 'orange')
        ax2.text(0.1, 0.1, 'Intensity = %.2E' %(grinty*10**exp),transform=ax2.transAxes,va='top', color='orange', bbox=dict(facecolor = 'white', alpha = 0.9, edgecolor='orange'))
        ax2.tick_params(axis='y', colors='orange')

        #save in csv
        with open('intensity.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([grstart, grinty*10**exp ])

        fig.tight_layout()
        fig.set_size_inches(20,6)
        plt.show()
    except:
        print(f'intentsity file for time: {filter} not found')


