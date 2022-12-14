'''
|--------------------------------------------
|Instructions to perform growth rate analysis
|--------------------------------------------
|
|- Helper files QMeter.py and sddsdata.py should be in the cwd
|- sdds files will be converted to parquet files by the script into *qmeter/MD3* folder (Only with Python 2 env)
|- Save manually the chromaticity trims from LSA into a .csv called *'chromaticity_trims.csv'* in the cwd
|   you can do this by selecting in the trim history in LSA, select the files to store, 
|   click on table, right click on any cell and select export. 
|- Plots with the linear fit will be saved into the *'plots/'* folder
|- Results will be saved into *'Growth_rate_and_chroma.csv'* and the plot into 'Growth_rate_and_chroma.png
|
| Jun 2022
|
|
'''
import os
import glob
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.signal
import csv
import pandas as pd
from datetime import datetime, timedelta
import math
from scipy.interpolate import interp1d
import csv

import QMeter
'''
# Run with python 2.7, activate it by
#$ source /afs/cern.ch/user/p/pyorbit/public/PyOrbit_env/virtualenvs/py2.7/bin/activate

# Convert mat files to parquet files

start_time = '2022_06_14 18:10:00'
end_time = '2022_06_14 19:00:00'
cycle = 'MD_26_L7200_Q26_North_Extraction_2022_V1'

QMeter.make_mat_files(start_time, end_time)
print('File conversion finished')
print('---------------------------------------------------------')

'''
# Run in python 3.7, activate it by
#$  . /user/spsscrub/2022/sps_beam_monitoring/setup_environment.sh

# Start STFT analysis

cwd = os.getcwd()+'/'
files = glob.glob(cwd+'qmeter/MD3/*')

for f in files:

    ni = 1000 #mask injection noise

    # Read file
    with open(f, 'rb') as fid:
        d = pickle.load(fid, encoding='latin1')

    # Retrieve data
    time_stamp = d['t_stamp_string']
    vertical_position_all = d['rawDataV']
    nf = len(vertical_position_all)
    if nf > 2**17: #for duplicated signals
        nf = nf//2
    vertical_position = d['rawDataV'][ni:nf]
    acqPeriod = d['acqPeriod']
    turns =np.arange(ni, nf, 1)

    # Start analysis
    if max(vertical_position) > 1e6:
        print('[PROGRESS] Analyzing file ' + time_stamp)

        # mask noise before instability by 2%
        mask_noise = abs(vertical_position) > max(vertical_position)*0.02
        igr = np.where(mask_noise)[0]
        #mask_noise[igr[0]-500:igr[-1]] = True

        # Perform STFT
        MWFFT = scipy.signal.stft(vertical_position[mask_noise], fs=1.0, window='hann', nperseg=256, noverlap=200, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
        #MWFFT = scipy.signal.stft(vertical_position, fs=1.0, window='hann', nperseg=256, noverlap=200, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
        
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8,11), num=time_stamp)
        ax1.set_xlabel('Number of turns')
        ax1.set_ylabel('Vertical position')
        ax1.plot(vertical_position_all, 'k')
        ax1.plot(turns[mask_noise], vertical_position[mask_noise], 'b') #masked noise

        # Obtain the log of the STFT amplitude
        log_peak_amplitude = []
        for i in range(len(MWFFT[1])):
            log_peak_amplitude.append(np.log(max(np.abs(MWFFT[2][:,i]))))

        log_peak_amplitude = np.array(log_peak_amplitude)
        ax3.set_xlabel('Number of STFT point')
        ax3.set_ylabel('log (Amplitude of Peak in FFT)')
        ax3.plot(MWFFT[1],log_peak_amplitude,'.-b', label='measurements')
        plt.tight_layout()

        # Mask log amplitude after maximum
        imax = np.argmax(log_peak_amplitude)
        mask_amp = log_peak_amplitude > 0
        mask_amp[imax+1:]=False

        # Mask log_pak_amplitude to get only values with positive slope
        df = MWFFT[1][2]-MWFFT[1][1]
        ddf = np.zeros_like(log_peak_amplitude)
        for i in range(2, len(MWFFT[1])-2):
            #fourth order scheme
            ddf[i+1] = (-log_peak_amplitude[i+2]+8*log_peak_amplitude[i+1]-8*log_peak_amplitude[i-1]+log_peak_amplitude[i-2])/(12*df)

        mask_pos=ddf > 0

        # Obtain final mask and plot
        mask = np.logical_and(mask_pos, mask_amp)
        gr = log_peak_amplitude[mask]
        points = MWFFT[1][mask]
        ifit = len(gr)-4

        if np.count_nonzero(mask) > 2.0:

            # Perform the first linear fit 
            coef, cov = np.polyfit(points, gr,1, cov=True)
            poly1d_fn = np.poly1d(coef)
            error= np.sqrt(np.diag(cov)[0])

            if abs(float(error)/float(coef[0])) > 0.1: #for bad signals, take only the last ifit points
                coef = np.polyfit(points[ifit:], gr[ifit:],1, cov=False)
                poly1d_fn = np.poly1d(coef)
                ax3.plot(points[ifit:] , gr[ifit:], '.-r', label='mask')
                ax3.plot(points[ifit:] ,poly1d_fn(points[ifit:]), '--k', label='fit masked data')

            else:
                ax3.plot(points , gr, '.-r', label='mask')
                ax3.plot(points ,poly1d_fn(points), '--k', lw=2.0, label='fit masked data')
            ax3.text(0.82,0.4," $\u03C4^{-1}$ = Slope = %.2E +/- %.E" %(coef[0], error),ha='right',va='top',transform=ax3.transAxes, color='black', fontsize = 15)

            slope = float(coef[0])
            uncertainty = float(error)
            nturns = len(vertical_position)
            plt.legend()

            # Save data in csv
            gr_title = 'growth_rate_values_14_06_2022_afternoon.csv'
            fi=open(cwd + gr_title, 'a')
            writer=csv.writer(fi, delimiter=',')
            writer.writerow([time_stamp, slope, uncertainty, nturns, acqPeriod])
            fi.close()
            
            # Save fig
            if not os.path.exists(cwd +'plots/'):
                os.mkdir(cwd +'plots/')
            plt.savefig(cwd +'plots/'+time_stamp +'.png')

            plt.close()

        else: print('[WARNING] File '+ time_stamp + ' contains no valid data for fit')
    else: print('[WARNING] File '+ time_stamp + ' contains no valid data for fit')

    #plt.show()

print('STFT analysis finished')
print('---------------------------------------------------------')

# Merge chromaticity trims with growth rate csv

# Get trims csv from LSA [TODO]
#import pjlsa
#lsa = pjlsa.LSAClient('sps')
#params = ['SPSBEAM/QPV']
#trims_raw = lsa.getTrims(parameter=params, cycle=cycle, start=start_time, end=end_time)
#print(trims_raw)

# Read datasets
qv_title = 'chromaticity_trims_afternoon.csv'
df_V = pd.read_csv(cwd+qv_title, header=None)
df_gr = pd.read_csv(cwd+gr_title, header=None)

# rearrange and sort data
df_V = df_V.drop(df_V.columns[range(0, df_V.shape[1], 2)], axis=1)
df_V = df_V.drop(1)
df_V.columns = range(df_V.shape[1])
df_gr.columns = range(df_gr.shape[1])
df_V = df_V.drop(range(8,12))
df_V = df_V.drop(range(2,7))
df_V = df_V.reset_index(drop=True)
df_V = df_V.transpose()
df_V = df_V.reset_index(drop=True)
df_V[0][0] = 'SPSBEAM/QPV (14-06-2022 13:50:00)'
df_V[0] = df_V[0].astype('str')
df_gr[0] = df_gr[0].astype('str')

# format time data
for i in range(0,len(df_gr[0])):
    aux = df_gr[0][i].split('_')
    df_gr[0][i] = aux[0] + '-' + aux[1] + '-' + aux[2] + ' ' + aux[3] + ':' + aux[4] + ':' + aux[5]

for ch in ["SPSBEAM/QPV ", "(", ")"]:
    for i in range(0,len(df_V[0])):
        df_V[0][i] = df_V[0][i].replace(ch,"")

df_V[0] = pd.to_datetime(df_V[0], dayfirst=True)
df_V.set_index(0, inplace=True)
df_gr[0] = pd.to_datetime(df_gr[0], dayfirst=True)
df_gr.set_index(0, inplace=True)

#increase the number of chromaticity values
df_V = df_V.resample('30S').ffill() 
df_V.reset_index(level=0, inplace=True)
df_V = df_V.rename(columns={0:'Time', 1:'QPV'})

df_gr.reset_index(level=0, inplace=True)
df_gr = df_gr.rename(columns={0:'Time', 1:'Growth_rate', 2:'error', 3:'turns', 4:'acqPeriod'})
df_gr = df_gr.sort_values(by=['Time'])

# merge in new dataset
new_df = pd.merge_asof(df_gr, df_V, on="Time", direction="nearest")
new_df = new_df.sort_values(by=['QPV'])

# obtain np.arrays 
QVf = (np.array(new_df['QPV'], dtype='float32')-0.13)*(-1.6)  # Chromaticity [GHz]
GR = np.array(new_df['Growth_rate'], dtype='float32')*1000    # Growth rate [1000/nturns]
nTurns = np.array(df_gr['turns'], dtype='float32')            # Number of turns used in STFT
error  = np.array(df_gr['error'], dtype='float32')*1000

# Sort
ii = np.argsort(QVf)
QVf = QVf[ii]
GR = GR[ii]
error = error[ii]
nTurns = nTurns[ii]

# Calibrate GR with number of turns used fpr STFT
#GR = (GR / nTurns) * nTurns[0]

# Obtain the mean of growth rate and error per trim
trims = np.unique(QVf)
GR_mean=np.zeros_like(trims)
error_mean = np.zeros_like(trims)
i = 0
for t in trims:
    mask = QVf == t
    GR_mean[i] = sum(GR[mask])/len(GR[mask])
    #error_mean[i] = np.std(GR[mask])+np.sqrt(sum((error[mask])**2)/len(error[mask]))
    error_mean[i] = np.std(GR[mask])+sum(error[mask])/np.sqrt(len(error[mask]))
    i+=1

xnew = np.linspace(trims.min(), trims.max(), 300)  
GR_smooth = interp1d(trims, GR_mean, kind='linear')
error_m = interp1d(trims, GR_mean-error_mean, kind='linear')
error_M = interp1d(trims, GR_mean+error_mean, kind='linear')

# plot
plt.figure(500)
plt.plot(QVf, GR, '.b', markersize=14)
plt.fill_between(xnew, error_m(xnew), error_M(xnew), alpha=0.7, facecolor='r')
plt.plot(xnew, GR_smooth(xnew), 'r')

plt.ylabel('Growth rate [1000/nturns]')
plt.xlabel('Chromaticity frequency [GHz]')
plt.title('Growth rate measurements 2022-06-14 - 3ns bunch')
plt.grid(alpha=0.5)

plt.show()

# save results
new_df.to_csv('Growth_rate_and_chroma_afternoon.csv')
plt.savefig('Growth_rate_and_chroma_afternoon.png')

print('Growth rate vs Chromaticity analysis finished')
print('---------------------------------------------------------')

