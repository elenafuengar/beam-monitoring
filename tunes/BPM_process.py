import glob
import datascout as ds
import harpy
import csv
import numpy as np
import pandas as pd
import os
import glob

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scienceplots
plt.style.use(['science','no-latex', 'ieee'])

from harpy.harmonic_analysis import HarmonicAnalysis

import warnings
warnings.filterwarnings("ignore")



def reshapeData(data):
        nbOfCapBunches = data['nbOfCapBunches']
        nbOfCapTurns = data['nbOfCapTurns']
        for par in ['horPosition','verPosition']:
            data[par]=[pos.reshape(nbOfCapTurns,nbOfCapBunches) for pos in data[par]]
        return data

def zero_crossings(array):
    beginning=next((i for i, x in enumerate(array) if x !=0 ), None)
    if 0 in array[beginning:beginning+30]:
        return False , 0
    else:
         return True , beginning

def save(fname='tuneshift_modes_lessharm.csv', dat=None):
    if dat == None:
        print('No data provided to save')
        return 1
       
    
        # write csv
    row=[]
    header=dat.keys()

    for key in header:
        row.append(dat[key])
    print('saving data',row)

    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(header)

    with open(fname, 'a') as f:  
        writer=csv.writer(f, delimiter=' ')
        writer.writerow(row)

def get_qpv_and_intensity(dict):
    
    intensity=None 
    qpv=None   # intensity
    try:
        intensity = dict['SPS.BCTDC.51456/Acquisition']['value']['totalIntensity']* \
        10**dict['SPS.BCTDC.51456/Acquisition']['value']['totalIntensity_unitExponent'] #[p/b]
        intms = dict['SPS.BCTDC.51456/Acquisition']['value']['measStamp'] # [ms]
    except:
        print(f'file contains no valid intensity data')

    cycle_offset=4015
    cycle_start=100

    X = dict['SPSBEAM/QPV']['value']['JAPC_FUNCTION']['X'] - cycle_offset
    Y = dict['SPSBEAM/QPV']['value']['JAPC_FUNCTION']['Y']
    qpv = float(Y[X > cycle_start][0])

    X = dict['SPSBEAM/QPH']['value']['JAPC_FUNCTION']['X'] - cycle_offset
    Y = dict['SPSBEAM/QPH']['value']['JAPC_FUNCTION']['Y']
    qph = float(Y[X > cycle_start][0])

    
    return qpv , np.mean(intensity[100:200])


def plot(df, fname='tuneshift_BPM.csv'):

    


    df=df[df['freq0'] > 0]
    df=df[df['freq0'] < 0.4]
    qpvs=np.unique(df['qpv'])
    entries=int((len(df.keys())-2)/2)


    for qpv in qpvs:
        fig, ax = plt.subplots(1,1,dpi=300, figsize=(3.25, 3.25))
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        fig.suptitle('Tune measurement', fontsize=8)
        ax.set_xlabel(r"Intensity e10")
        ax.set_ylabel(r'Tune')

        if qpv >10:
            continue

        intensities=df.loc[df['qpv'] == qpv, 'intensity']
        
        for i in range(entries):
            ampl_max=np.max(np.abs(df.loc[df['qpv'] == qpv, f'ampl{i}']))
            mode=df.loc[df['qpv'] == qpv, f'freq{i}']
            amplitude=df.loc[df['qpv'] == qpv, f'ampl{i}']
            
            bol = np.logical_and(intensities > 3e10 , intensities < 30e10)    
           

            if i == 0:
                color = 'r'
                coef, cov = np.polyfit(intensities[bol],mode[bol], 1, cov=True)
                trendpoly = np.poly1d(coef)
                ax.plot(intensities[bol]/1e10,trendpoly(intensities[bol]), color='r', label=f'Qshift={coef[0]:03e}')
            else:
                color = 'k'
            ax.scatter( intensities[bol]/1e10 , mode[bol] ,s=0.2,c=color,alpha=np.abs(amplitude[bol])/ampl_max,)
            
            ax.set_ylim(top=0.28)
            plt.legend(frameon=True,prop={'size': 4})
            #plt.show()
            plt.savefig(f'modes_qpv{qpv}_test.png')


#line2=ax.plot(xaxis , slopes2 , 'k', label='Tuneshift simulation (PyHT)')






n_harm=4


measurements=glob.glob('/eos/user/m/miguelgo/joined/*.parquet')

freq_col=[ f'freq{i}' for i in range(n_harm)]
ampl_col=[ f'ampl{i}' for i in range(n_harm)]


df = pd.DataFrame(columns=['qpv','intensity']+freq_col+ampl_col)

for file in measurements[230:]:
    print(f'Analizing {file[40:]}')
    
    hour = int(file.split('/')[-1].split('.')[3])
    minute = int(file.split('/')[-1].split('.')[4])
    if hour < 14:
        print('skipped hour',hour)
        continue
    dat=ds.parquet_to_dict(file)
    try:
        qpv , intensity = get_qpv_and_intensity(dat)
    except:
        print(f'skipped {file[len(file)-20:]}')
        continue
    if qpv < 0.1:
        print(qpv)
        continue
    row=[None]*(n_harm*2+2)
    row[0]=qpv
    row[1]=intensity
    #We are only looking at BPM2
    for monitor_ID in [1]:#range(3):
        try:
        
            batches=reshapeData(dat['BPLOFSBA5/GetCapData']['value'])['verPosition'][monitor_ID]
            msk = np.where(np.mean(batches,axis=0))[0]
            if not msk.size >0 :
                print('No data in this BPM')
                continue
            
            signal=batches[:,msk[0]]
            kick_time = 65 #you have to look at it manually, usually constant in an md

            fig, ax = plt.subplots(1,1,dpi=300, figsize=(3.25, 3.25))
            fig.tight_layout(rect=[0, 0.12, 1, 1])
            fig.suptitle(f'BPM Signal at qpv{qpv}', fontsize=8)
            ax.set_xlabel(r"BPM turns")
            ax.set_ylabel(r'Pos')
            x = range(len(signal))
            ax.plot(signal, c='k')
            ax.axvline(kick_time, c='b')
            ax.axvline(kick_time+16, c='b')
            fig.savefig(f'BPMsignals/BPM_signal_qpv{qpv}_I{intensity:.2e}_time{hour}-{minute}.png')


            flag , start = True, kick_time #zero_crossings(signal)
            

            if flag:
                
                morsel=signal[start : start+16]
                morsel=morsel-np.mean(morsel)
                

                ha=HarmonicAnalysis(morsel)

                ### Laskar method returns in secon place in the array a 'recoprocal' of the true tune 
                # e.g. if the tune is 0.21 it will return [0.21, 0.79, (rest of the harmonics) ] while analyzing a real signal
                #In plotBPM.py it takes the minimum of the first two returned frequencies. 
                
                full=ha.laskar_method(num_harmonics=n_harm) 
                row[2:12]=full[0]
                row[12:]=np.real(full[1])
                
                
                
                #do tune computation and write it to the appropriate tune
        
        except:
            continue
        df.loc[len(df)] = row

        print(df)
df.to_csv('BPM_processed_testi.csv', sep='\t')
#plot(df)










    


