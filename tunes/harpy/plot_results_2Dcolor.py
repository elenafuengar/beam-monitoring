import h5py as h5
import matplotlib.pylab as plt
plt.rcParams.update({'font.size': 20})
import numpy as np
from harpy.harmonic_analysis import HarmonicAnalysis
import os
from collections import deque
from scipy.optimize import curve_fit
from lmfit import Model
import plotly.express as px
import scipy.signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

plt.close('all')
plt.ion()
#loading input data
fhandle = h5.File("afs/cern.ch/user/i/imasesso/Desktop/harpy/test_tuneharmonics//bunchmonitor_xi120000000000.0.h5")


hnum=40 #number of modes taken into account
indexT=64 #change number of turns analyzed
caseUT='fullmodel/singlebunch/motion'
nturns_2_analyzevec = np.arange(16, 1000, 1)
nturns_2_analyze=nturns_2_analyzevec[indexT]
nturnsstr=["%03d" % nturns_2_analyze]
plane = 'x' #plane switch

#nmp=2e5; #number of macroparticle per bunch
#nmpstr=str(nmp)


b_index = []
arr = []
arrm=[]
arrE=[]
tunes=[]

for b in sorted([int(i) for i in list(fhandle['Bunches'])]):
    b_index.append(b)
    arr.append(fhandle['Bunches'][str(b)]['mean_%s'%plane][:])
    arrm.append(fhandle['Bunches'][str(b)]['macroparticlenumber'][:])
    arrE.append(fhandle['Bunches'][str(b)]['epsn_%s'%plane][:])

    #using harpy for tune analysis
    ha = HarmonicAnalysis(arr[-1][:nturns_2_analyze])
    tunes.append(ha.laskar_method(num_harmonics=hnum)[0][0])
    
tunes=np.abs(np.array(tunes)-np.round(tunes))

arr = np.array(arr)
arrm = np.array(arrm)
arrE = np.array(arrE)

bnum=0; #bunch number for FFT window analysis

f, ax = plt.subplots(1)
ax.set_xlabel('turn',fontsize=25)
ax.set_ylabel('mean %s [m]'%plane,fontsize=25)
ax.plot(arr.T)
#ax.set_ylim(-0.1,0.1)
plt.ticklabel_format(axis='both', style='sci', scilimits=(-4,4))
plt.tight_layout()


#### Moving window FFT starts here
'''
dh=arr[bnum]
MWFFT=scipy.signal.stft(dh, fs=1.0, window='hann', nperseg=128, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
MWFFT2=np.transpose(np.array(MWFFT[2]))
i=range(1,len(MWFFT[1]))
iss=str(i)

import colorsys
N = len(MWFFT[0])
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

f, ax = plt.subplots(1)
ax.set_xlabel('',fontsize=25)
ax.set_ylabel('SFFT',fontsize=25)
#ax.plot(MWFFT[0],np.abs(MWFFT[2]), label='data')
ax.plot(MWFFT[0],np.abs(MWFFT[2]), label='data')

ax.set_ylim(-5e-5,5e-5)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-4,4))
plt.legend(fontsize=10)
plt.tight_layout()
#plt.show()

'''
    #Moving Window ends here

#additional plots
'''
f, ax = plt.subplots(1)
ax.set_xlabel('turn',fontsize=25)
ax.set_ylabel('Losses %',fontsize=25)
ax.plot((nmp-arrm.T)/nmp*100)
#ax.set_ylim(-0.1,0.1)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(0,6))
plt.tight_layout()

f, ax = plt.subplots(1)
ax.set_xlabel('turn',fontsize=25)
ax.set_ylabel('epsn %s [m]'%plane,fontsize=25)
ax.plot(arrE.T)
ax.set_ylim(2e-6,4.5e-6)
plt.ticklabel_format(axis='both', style='sci',scilimits=(0,6))
plt.tight_layout()

AmpMax=np.nanmax(arr,axis=1)
b_index2plot = np.arange(max(b_index)+1)
Amp2plot = np.zeros_like(b_index2plot) * np.nan
Amp2plot[b_index] = AmpMax

LossMax=np.nanmin(arrm,axis=1)
b_index2plotL = np.arange(max(b_index)+1)
Loss2plot = np.zeros_like(b_index2plotL) * np.nan
Loss2plot[b_index] = (nmp-LossMax)/nmp*100

ermsMax=np.divide((np.nanmax(arrE,axis=1)-np.nanmin(arrE,axis=1)),np.nanmin(arrE,axis=1))*100
b_index2plotL = np.arange(max(b_index)+1)
erms2plot = np.zeros_like(b_index2plotL) * np.nan
erms2plot[b_index] = ermsMax


f, ax = plt.subplots(1)
ax.set_xlabel('bunch',fontsize=25)
ax.set_ylabel('max %s [m]'%plane,fontsize=25)
ax.semilogy(b_index2plot, Amp2plot,'ro')
#ax.set_ylim(-0.001,0.001)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
plt.tight_layout()

f, ax = plt.subplots(1)
ax.set_xlabel('bunch',fontsize=25)
ax.set_ylabel('Losses %',fontsize=25)
ax.semilogy(b_index2plotL, Loss2plot,'ro')
#ax.set_ylim(-0.001,0.001)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
plt.tight_layout()

f, ax = plt.subplots(1)
ax.set_xlabel('bunch',fontsize=25)
ax.set_ylabel('epsn %s [m]'%plane,fontsize=25)
ax.semilogy(b_index2plotL, erms2plot,'ro')
#ax.set_ylim(-0.001,0.001)
#plt.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
plt.tight_layout()

''

b_index2plot = np.arange(max(b_index)+1)
tunes2plot = np.zeros_like(b_index2plot) * np.nan
tunes2plot[b_index] = tunes

f, ax = plt.subplots(1)
ax.set_xlabel('bunch',fontsize=25)
ax.set_ylabel('Q%s'%plane,fontsize=25)
ax.plot(b_index2plot, tunes2plot,'bo')
plt.tight_layout()
'''
#End additional plots

plane = 'x'
b_index = []
arr = []
arrm=[]
arrE=[]
tunesx = []
tunesy = []
ampx= []
ampy= []

for b in sorted([int(i) for i in list(fhandle['Bunches'])]):
    b_index.append(b)
    arr.append(fhandle['Bunches'][str(b)]['mean_%s'%plane][:])
    arrm.append(fhandle['Bunches'][str(b)]['macroparticlenumber'][:])
    arrE.append(fhandle['Bunches'][str(b)]['epsn_%s'%plane][:])

    #using harpy for tune analysis
    ha = HarmonicAnalysis(arr[-1][:nturns_2_analyze])
    tunesx.append(ha.laskar_method(num_harmonics=hnum)[0])
    ampx.append(ha.laskar_method(num_harmonics=hnum)[1])

plane = 'y'
b_index = []
arr = []
arrm=[]
arrE=[]

for b in sorted([int(i) for i in list(fhandle['Bunches'])]):
    b_index.append(b)
    arr.append(fhandle['Bunches'][str(b)]['mean_%s'%plane][:])
    arrm.append(fhandle['Bunches'][str(b)]['macroparticlenumber'][:])
    arrE.append(fhandle['Bunches'][str(b)]['epsn_%s'%plane][:])

    #using harpy for tune analysis
    ha = HarmonicAnalysis(arr[-1][:nturns_2_analyze])
    tunesy.append(ha.laskar_method(num_harmonics=hnum)[0])
    ampy.append(ha.laskar_method(num_harmonics=hnum)[1])

b_index2plot = np.arange(max(b_index)+1)
tunes2plotx = np.zeros_like(b_index2plot) * np.nan
tunes2ploty = np.zeros_like(b_index2plot) * np.nan

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}


f, ax = plt.subplots(1, figsize=(6.4*1.4,4.8))
ax.set_ylabel('Horizontal tune',fontsize=20)
ax.set_xlabel('Bunch number',fontsize=20)
#ax.plot(MWFFT[0],np.abs(MWFFT[2]), label='data')
x=np.transpose(np.matlib.repmat(np.linspace(0,71,72),hnum,1))
#y=np.linspace(1,1000,1000)
yx=[]
yy=[]
for pp in range(72):
    for ii in range(hnum):
        yx0=np.abs(tunesx[pp][ii]-round(tunesx[pp][ii]))
        yy0=np.abs(tunesy[pp][ii]-round(tunesy[pp][ii]))
        yx.append(yx0)
        yy.append(yy0)

#y=np.transpose(y)
yx=(np.reshape(yx,[72,hnum]))
yy=(np.reshape(yy,[72,hnum]))
i=np.arange(0,1000,1)
ip=i+1
#z=np.abs((ampx))
zx=20*np.log10(np.abs((ampx))/np.max(np.abs((ampx))))
zy=20*np.log10(np.abs((ampy))/np.max(np.abs((ampy))))
#z=20*np.log10(np.divide((np.abs(MWFFT2[ip]-MWFFT2[i])), np.max(np.abs(MWFFT2[ip]-MWFFT2[i]))))

#ax.plot(x,y,'o')
#ax.scatter(x,y,z)
#plt.show()

#Xplane
i_sorted = np.argsort(zx.flatten())[::1] 
x_sorted = x.flatten()[i_sorted]
y_sorted = yx.flatten()[i_sorted]
z_sorted = zx.flatten()[i_sorted]

#Yplane
#i_sorted = np.argsort(zy.flatten())[::1]
#x_sorted = x.flatten()[i_sorted]
#y_sorted = yy.flatten()[i_sorted]
#z_sorted = zy.flatten()[i_sorted]

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
im=ax.scatter(x_sorted,y_sorted,c=z_sorted, cmap='gray_r', vmin=-18, vmax=0)
ax.plot(tunes,'r.',label='Computed tune')
cbar = plt.colorbar(im, cax=cax)    
cbar.ax.set_ylabel('Relative FFT Amp. [dB]')
#ax.set_ylim(0.22,0.30)
ax.set_ylim(0.10,0.18)

ax.set_title('number of turns analyzed = '+nturnsstr[0]+'.')

plt.subplots_adjust(bottom=0.165, right=0.80)
plt.savefig('Results/'+caseUT+'/HVtune72nturns'+nturnsstr[0]+'.png')
plt.show()





Datawrite = np.array([b_index2plot, tunes2plotx, tunes2ploty])
Dataerms = np.array([b_index2plotL, erms2plot])
np.savetxt('Results/'+caseUT+'/HVtune72nturns'+nturnsstr[0]+'.txt', Datawrite.T, delimiter='\t')
np.savetxt('Results/'+caseUT+'/slowermstest'+nmpstr+'.txt', Dataerms.T, delimiter='\t')
  


