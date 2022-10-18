import datascout as ds
import numpy as np
import os, glob
from harpy.harmonic_analysis import HarmonicAnalysis
import matplotlib.pyplot as plt
import csv 

path ='/usex = np.array(median_tunesx)
tunesy = np.array(median_tunesy)
idy=np.isfinite(tunesy)
idx=np.isfinite(tunesx)

px, covx = np.polyfit(DpOverP[idx]/1000, tunesx[idx],1, cov = True)
py, covy = np.polyfit(DpOverP[idy]/1000, tunesy[idy],1, cov = True)
/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/chromaticity_measurments/SPS.USER.MD3/2022.08.15/'
folder = 'QPH0.1_QPV0.1_N3e10'
parquet_list=sorted(glob.glob(path+folder+'/*.parquet'))

#parquet_list=glob.glob(path+folder+'/'+'2022.07.26.17.00.28.141007.parquet')

threshold = 1e10 #intensity threshold to skip files with no beam
cycle_offset = 1015
cycle_start = 200
cycle_end = 3000

QPV_time=[]
QPH_time=[]
RadialSteering_time = []
DpOverP_time = []
RevFreq_time = []
Tunes = []

def get_LSA_time_value(dict, cycle_offset=0, cycle_start=cycle_start, cycle_end=cycle_end):        
	X = dict['value']['JAPC_FUNCTION']['X'] - cycle_offset
	Y = dict['value']['JAPC_FUNCTION']['Y']
	vals = Y[X > cycle_start]
	return float(vals[0])

def get_BCT(dict): #TO DO
	TotalIntensity = np.array(dict['value']['totalIntensity'])
	exp = np.array(dict['value']['totalIntensity_unitExponent'])
	
	return TotalIntensity*10**exp

def get_RevFreq(dict):
	vals = dict['value']['data']
	return float(vals[10000])
	
def get_BBQ_tunes(dict):
	
	msTags = dict['value']['msTags']
	frevfreq = np.mean(dict['value']['frevFreq'])
	#Auro Q app values
	delay_ms = 50
	start_ms = 200 + delay_ms 
	stop_ms = 3000 + delay_ms 
	interval_ms = 20

	stop_turns = msTags[range(start_ms,stop_ms,interval_ms)[1:]]
	start_turns = msTags[range(start_ms,stop_ms,interval_ms)[:-1]]
	ms_arr = (start_turns+stop_turns) / frevfreq 
		
	tunes = {'H': [], 'V': []}
	median_tunes = {'H': [], 'V':[]}
	average_tunes = {'H': [], 'V':[]}
	
	for plane in tunes:
		#plt.figure()
		for i in range(len(stop_turns)):
			pos=dict['value']['rawData'+plane][start_turns[i]:stop_turns[i]]
			'''
			if i < 9: #plot raw position to check the chirp
				plt.subplot(3,3,i+1)
				plt.plot(pos)
				plt.suptitle('Raw position plane ' + plane)
			'''
			analysis=HarmonicAnalysis(pos)
			tune_modes = analysis.laskar_method(num_harmonics=40)[0]
			tune_modes = np.abs(np.array(tune_modes)-np.round(np.array(tune_modes)))
			tune_mode_0 = tune_modes[0]
                   
			if plane == 'H':
				if tune_mode_0[tune_mode_0 > 0.1 and tune_mode_0 < 0.17].size > 0:
					tunes[plane].append(float(tune_mode_0[tune_mode_0 > 0]))
				else: 
					tunes[plane].append(np.nan)
			if plane == 'V':
				if tune_mode_0[tune_mode_0 > 0.14].size > 0:
					tunes[plane].append(float(tune_mode_0[tune_mode_0 > 0]))
				else:
					tunes[plane].append(np.nan)

		mask = np.logical_and(ms_arr > 200, ms_arr < 3000)
		tunes[plane]=np.array(tunes[plane])
		tunes[plane]=tunes[plane][mask]
		median_tunes[plane] = np.nanmedian(tunes[plane])
		average_tunes[plane] = np.nanmean(tunes[plane])
	#plt.show()
	return {'tunes' : tunes, 'median_tunes':median_tunes, 'average_tunes':average_tunes, 'ms':ms_arr[mask]}

# Start file analysis
print('Start analysis for data in '+folder+'/:')

for parquet in parquet_list:
	print('    - analysing ' + parquet.split('/')[-1] + ' file...')
	data = ds.parquet_to_dict(parquet)
	filename = parquet.split('/')[-1]
	time = filename.split('.parquet')[0]

	BBQCONT = data['SPS.BQ.CONT/ContinuousAcquisition']
	QPV = data['SPSBEAM/QPV']
	QPH = data['SPSBEAM/QPH']
	RadialSteering = data['SpsLowLevelRF/RadialSteering']
	DpOverPOffset = data['SpsLowLevelRF/DpOverPOffset']
	BCT = data['SPS.BCTDC.41435/Acquisition']
	RevFreq = data['SA.RevFreq-ACQ/Acquisition']
	#skip files with low intensity
	try:
		Intensity = get_BCT(BCT)
		if np.max(Intensity) > threshold:
		
			QPV_time.append(get_LSA_time_value(QPV))
			QPH_time.append(get_LSA_time_value(QPH))
			RadialSteering_time.append(get_LSA_time_value(RadialSteering, 1000))
			DpOverP_time.append(get_LSA_time_value(DpOverPOffset, 1000))
			RevFreq_time.append(get_RevFreq(RevFreq))
			Tunes.append(get_BBQ_tunes(BBQCONT))
	except:
		pass


# Recover time arrays
	
QPV = np.array(QPV_time)
QPH = np.array(QPH_time)
RadialSteering = np.array(RadialSteering_time)
DpOverP = np.array(DpOverP_time)
RevFreq = np.array(RevFreq_time)

# Plot tunes for 1 file

tunesx = Tunes[0]['tunes']['H']
tunesy = Tunes[0]['tunes']['V']
time = Tunes[0]['ms']

fig, ax = plt.subplots()
ax.scatter(time, tunesx, label='H')
ax.scatter(time, tunesy, label='V')
ax.set_xlabel('time [ms]')
ax.set_ylabel('Tune Q')
plt.legend()

# Plot median tunes for all the files
fig, (ax1, ax2) = plt.subplots(1, 2)
for t in range(len(Tunes)):
	ax1.scatter(t, Tunes[t]['median_tunes']['H'])
	ax1.set_xlabel('File No. # ')
	ax1.set_ylabel('Median tunes Q')
	ax1.set_title('Horizontal tunes')

	ax2.scatter(t, Tunes[t]['median_tunes']['V'])
	ax2.set_xlabel('File No. # ')
	ax2.set_ylabel('Median tunes Q')
	ax2.set_title('Vertical tunes')

fig.tight_layout()

	
#Plot tune vs DpOverP
fig, (ax1, ax2) = plt.subplots(1, 2)
median_tunesx=[]
median_tunesy=[]
'''
DpOverP=np.append(0.0,DpOverP) #AutoQ app and acquisition are not in sync
DpOverP=np.delete(DpOverP,len(DpOverP)-1)
DpOverP[np.where(DpOverP==-0.001)[0][0]-1]=-0.001
DpOverP=DpOverP*1000
'''
# Formula with slip factor for Q20 only
# DpOverPmeas=-(RevFreq-43347.2890625)/43347.2890625/1.8e-3*1000 #dp/p = dw/w/slipfactor
DpOverPmeas=-(RevFreq-43347.2890625)/max(RevFreq-43347.2890625)
DpOverP=DpOverPmeas #Dp/P obtained from the RevFreq acquisition
print(DpOverP)
for t in range(len(Tunes)):
	ax1.scatter(DpOverP[t], Tunes[t]['median_tunes']['H'])
	ax1.set_xlabel('dp/p [permill]')
	ax1.set_ylabel('Median tunes Q')
	ax1.set_title('Horizontal tunes vs dp/p offset')

	ax2.scatter(DpOverP[t], Tunes[t]['median_tunes']['V'])
	ax2.set_xlabel('dp/p [permill] ')
	ax2.set_ylabel('Median tunes Q')
	ax2.set_title('Vertical tunes vs dp/p offset')
	
	median_tunesx.append(float(Tunes[t]['median_tunes']['H']))
	median_tunesy.append(float(Tunes[t]['median_tunes']['V']))

tunesx = np.array(median_tunesx)
tunesy = np.array(median_tunesy)
idy=np.isfinite(tunesy)
idx=np.isfinite(tunesx)

px, covx = np.polyfit(DpOverP[idx]/1000, tunesx[idx],1, cov = True)
py, covy = np.polyfit(DpOverP[idy]/1000, tunesy[idy],1, cov = True)

Px=np.poly1d(px)
Py=np.poly1d(py)

ax1.plot(DpOverP, Px(DpOverP/1000),c='r', ls='-', label='fit')
ax2.plot(DpOverP, Py(DpOverP/1000),c='r', ls='-', label='fit')

QPx = float(px[0]/20)
QPy = float(py[0]/20)
errorQPx = np.sqrt(np.diag(covx)[0])
errorQPy = np.sqrt(np.diag(covy)[0])

ax1.text(0.1,0.8, 'fitted QPH = '+ str(round(QPx,2)), transform=ax1.transAxes, \
         bbox=dict(boxstyle='round',facecolor='white',alpha=0.5), fontsize=10 )
ax2.text(0.1,0.8, 'fitted QPV = '+ str(round(QPy,2)), transform=ax2.transAxes, \
         bbox=dict(boxstyle='round',facecolor='white',alpha=0.5), fontsize=10 )

plt.suptitle('Set ' + folder, y=0.95) 
fig.tight_layout()
plt.show()

# save data and fig  
fig.savefig(path+'plots/'+folder+'.png')

fname='chromaticity_values_corrected.csv'

header=['set', 'QPH','QPV']
if not os.path.exists(path+fname):
	f=open(path+fname, 'w')
	writer = csv.writer(f)
	writer.writerow(header)
	f.close()

row = [folder, QPx, QPy]
with open(path+fname, 'a') as f:
	writer = csv.writer(f)
	writer.writerow(row)

