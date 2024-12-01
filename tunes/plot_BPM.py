import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex', 'ieee'])




fig, ax = plt.subplots(1,1,dpi=300, figsize=(3.25, 3.25))
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.suptitle('Tune measurement', fontsize=8)
ax.set_xlabel(r"Intensity e10")
ax.set_ylabel(r'Tune')


sims_path='BPM_processed.csv'


df = pd.read_csv( sims_path , sep='\t')#.sort_values(by='qpv')
    


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
    ampl_max=np.max(np.abs(df.loc[df['qpv'] == qpv, f'ampl0']))
    for i in range(entries):
        mode=df.loc[df['qpv'] == qpv, f'freq{i}']
        amplitude=df.loc[df['qpv'] == qpv, f'ampl{i}']
            
        bol = np.logical_and(intensities > 5e10 , intensities < 30e10)    
           

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

#plt.legend(frameon=True,prop={'size': 4})

