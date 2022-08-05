import datascout as ds
import numpy as np
import os, glob
from harpy.harmonic_analysis import HarmonicAnalysis
import matplotlib.pyplot as plt
import csv 
import time

# LSA function settings
cycle_offset = 1015
cycle_start = 200
cycle_end = 3000

# Initialize
QPV_time=[]
QPH_time=[]
RadialSteering_time = []
DpOverP_time = []
RevFreq_time = []
Tunes = []

# helper functions
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
	
	#Auto Q app values
	global delay_ms
	global start_ms 
	global stop_ms
	global interval_ms

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
			if tune_mode_0[tune_mode_0 > 0.1].size > 0:
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

def run_analysis(parquet):
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
	
	global QPV_time
	global QPH_time
	global RadialSteering_time
	global DpOverP_time
	global RevFreq_time
	global Tunes

	#skip files with low intensity
	try:
		Intensity = get_BCT(BCT)
		if np.max(Intensity) > 1e10:

			QPV_time.append(get_LSA_time_value(QPV))
			QPH_time.append(get_LSA_time_value(QPH))
			RadialSteering_time.append(get_LSA_time_value(RadialSteering, 1000))
			DpOverP_time.append(get_LSA_time_value(DpOverPOffset, 1000))
			RevFreq_time.append(get_RevFreq(RevFreq))
			Tunes.append(get_BBQ_tunes(BBQCONT))
	except:
		pass

def plot_tunes_vs_dp():
	'''
	Plot meadian tune vs DpOverP measured from the RevFreq acquisition
	'''
	fig, (ax1, ax2) = plt.subplots(1, 2)	
	median_tunesx=[]
	median_tunesy=[]

	global RevFre_time
	RevFreq = np.array(RevFreq_time)

	global DpOverPmeas
	DpOverPmeas=-(RevFreq-43347.2890625)/max(RevFreq-43347.2890625)  

	global Tunes
	for t in range(len(Tunes)):
		ax1.scatter(DpOverPmeas[t], Tunes[t]['median_tunes']['H'])
		ax1.set_xlabel('dp/p [permill]')
		ax1.set_ylabel('Median tunes Q')
		ax1.set_title('Horizontal tunes vs dp/p offset')

		ax2.scatter(DpOverPmeas[t], Tunes[t]['median_tunes']['V'])
		ax2.set_xlabel('dp/p [permill] ')
		ax2.set_ylabel('Median tunes Q')
		ax2.set_title('Vertical tunes vs dp/p offset')

		median_tunesx.append(float(Tunes[t]['median_tunes']['H']))
		median_tunesy.append(float(Tunes[t]['median_tunes']['V']))

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()

def fit_tunes_vs_dp():
	'''
	Plot meadian tune vs DpOverP measured from the RevFreq acquisition
	'''
	fig, (ax1, ax2) = plt.subplots(1, 2)
	median_tunesx=[]
	median_tunesy=[]
	global folder

	global RevFreq_time
	RevFreq = np.array(RevFreq_time)

	global DpOverPmeas
	DpOverPmeas=-(RevFreq-43347.2890625)/max(RevFreq-43347.2890625)
	
	global Tunes
	for t in range(len(Tunes)):
		ax1.scatter(DpOverPmeas[t], Tunes[t]['median_tunes']['H'])
		ax1.set_xlabel('dp/p [permill]')
		ax1.set_ylabel('Median tunes Q')
		ax1.set_title('Horizontal tunes vs dp/p offset')

		ax2.scatter(DpOverPmeas[t], Tunes[t]['median_tunes']['V'])
		ax2.set_xlabel('dp/p [permill] ')
		ax2.set_ylabel('Median tunes Q')
		ax2.set_title('Vertical tunes vs dp/p offset')

		median_tunesx.append(float(Tunes[t]['median_tunes']['H']))
		median_tunesy.append(float(Tunes[t]['median_tunes']['V']))
	
	px, covx = np.polyfit(DpOverPmeas/1000, np.array(median_tunesx),1, cov = True)
	py, covy = np.polyfit(DpOverPmeas/1000, np.array(median_tunesy),1, cov = True)

	Px=np.poly1d(px)
	Py=np.poly1d(py)

	ax1.plot(DpOverPmeas, Px(DpOverPmeas/1000),c='r', ls='-', label='fit')
	ax2.plot(DpOverPmeas, Py(DpOverPmeas/1000),c='r', ls='-', label='fit')

	QPx = float(px[0]/20)
	QPy = float(py[0]/20)
	errorQPx = np.sqrt(np.diag(covx)[0])/20
	errorQPy = np.sqrt(np.diag(covy)[0])/20

	ax1.text(0.1,0.8, 'fitted QPH = '+ str(round(QPx,2)), transform=ax1.transAxes, \
		 bbox=dict(boxstyle='round',facecolor='white',alpha=0.5), fontsize=10 )
	ax2.text(0.1,0.8, 'fitted QPV = '+ str(round(QPy,2)), transform=ax2.transAxes, \
		 bbox=dict(boxstyle='round',facecolor='white',alpha=0.5), fontsize=10 )

	plt.suptitle('Set ' + folder, y=0.95, fontweight='bold')

	fig.tight_layout()
	plt.show()
	
	# save fig  
	fig.savefig(path+'plots/'+folder+'.png')
	
	# save data in csv row
	fname='new_chromaticity_values.csv'

	header=['set', 'QPH','QPV', 'errorQPH', 'errorQPV']
	if not os.path.exists(path+fname):
		f=open(path+fname, 'w')
		writer = csv.writer(f)
		writer.writerow(header)
		f.close()

	row = [folder, QPx, QPy, errorQPx, errorQPy]
	with open(path+fname, 'a') as f:
		writer = csv.writer(f)
		writer.writerow(row)

def move_files_to_folder(files, folder):
	global path
	if not os.path.exists(path+folder):
		os.mkdir(path+folder)
	for file in files:
		os.replace(file, path+folder+'/'+file.split('/')[-1])


#------------------------------------------------------------------------
# Set up
path ='/user/spsscrub/2022/sps_beam_monitoring/sps_beam_monitoring/data/chromaticity_measurments/SPS.USER.MD5/2022.08.02/'
folder = 'QPH0.3_QPV1_KLOF0_N10e10'

#Auto Q app values
delay_ms = 50
start_ms = 200 + delay_ms
stop_ms = 3000 + delay_ms
interval_ms = 20

# chromaticity settings in Auto Q
dp_rep = 3
dp_start = -1
dp_end = 1
dp_step = 0.5

# LSA function settings
cycle_offset = 1015
cycle_start = 200
cycle_end = 3000

print('Start analysis:')

parquet_list=sorted(glob.glob(path+'*.parquet'), key=os.path.getmtime)
parquet = parquet_list[-1]
parquets_analyzed = []
count_max = ((dp_end-dp_start)/dp_step+1)*dp_rep
count = 0
plt.ion()

while count < dp_rep:
	
	run_analysis(parquet)
	plt.close()
	plot_tunes_vs_dp()
	
	if DpOverPmeas[-1] == dp_end:
		count +=1
	
	while True:
		parquet_list=sorted(glob.glob(path+'*.parquet'), key=os.path.getmtime)
		if parquet_list[-1] != parquet:
			parquets_analyzed.append(parquet)
			parquet = parquet_list[-1]
			time.sleep(2)
			break
plt.close()
fit_tunes_vs_dp()
move_files_to_folder(parquets_analyzed, folder)

print('Analysis finished')
#-----------------------------------------------------------------------
