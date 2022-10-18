'''BQHT headtail monitor file analysis

Sep 22
'''

import glob
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

path = '/nfs/cs-ccr-bqhtnfs/sps_data/SPS.BQHT/2022_09_05/'
files = sorted(glob.glob(path+'*.h5'))
#filename = 'SPS.BQHT_MD3_20220905_154334.h5' #mode 0
#filename = 'SPS.BQHT_MD3_20220905_164934.h5' #mode 1
#filename = 'SPS.BQHT_MD3_20220905_165010.h5'
filename = 'SPS.BQHT_MD3_20220905_165158.h5'

files = [filename]
ifiles = []

Stored_turns = 1000 #SPS HT Control Settings: Number of turns saved
Start_turns = 160
End_turns = 190
Freq_turns = 1 #plot data every freq_turns

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Blues(np.linspace(0,1, int( (End_turns - Start_turns)/ Freq_turns) )))

for file in files:
    with h5.File(file, 'r') as hf:
        delta_H = np.array(hf.get('horizontal/delta/'))
        delta_V = np.array(hf.get('vertical/delta/'))
    
    len_delta_H = len(delta_H)
    len_delta_V = len(delta_V)

    len_x1 = int(len_delta_H / Stored_turns)
    len_x2 = int(len_delta_V / Stored_turns)

    x1 = np.arange(0,len_x1)
    x2 = np.arange(0,len_x2)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(Start_turns,End_turns,Freq_turns):
        ax1.plot(x1,delta_H[len_x1*i:len_x1*(i+1)])
        ax2.plot(x2,delta_V[len_x2*i:len_x2*(i+1)])

    #plt.colorbar()
    ax2.text(0.02, 0.1, f'Number of turns: {End_turns-Start_turns}', transform=ax2.transAxes,va='top')
    ax2.text(0.02, 0.05, f'Plotted every {Freq_turns} turns', transform=ax2.transAxes,va='top')
    ax1.set_title('Horizontal delta', fontweight='bold')
    ax2.set_title('Vertical delta $\delta_v$', fontweight='bold')
    ax1.set_xlim(240,360)
    ax2.set_xlim(240,360)
    ax1.set_ylim(min(delta_V)*1.2, max(delta_V)*1.2)
    ax2.set_ylim(min(delta_V)*1.2, max(delta_V)*1.2)
    ax2.set_xlabel('z position [a.u.]')
    ax2.set_ylabel('Position [a.u.]')
    fig.set_size_inches(14,5)
    ax2.grid(which='both', alpha =0.5, linestyle='dotted')
    fig.suptitle('Head Tail viewer - ' + filename) 
    plt.show()
