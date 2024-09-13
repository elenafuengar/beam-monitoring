'''
Script to launch PyHEADTAIL simualtions 
with single bunch for the CERN SPS with Q26 
optics, and calculate the HeadTail mode zero 
instability growth rates with different 
chromaticities

Date: 21/02/23
Author: edelafue
'''

import sys, os
import time

print(os.getcwd())

import numpy as np
import h5py
from scipy.constants import c, e, m_p
import matplotlib.pyplot as plt

from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeField, CircularResonator, Resonator, WakeTable
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.machines.synchrotron import Synchrotron
from SPSoctupolesNewConfiguration import SPSOctupolesNew
import PyHEADTAIL.aperture.aperture as aperture

from tqdm import tqdm

import glob
import shutil



def run(qpv_val, n_slices, n_mp, intensity):

    # BEAM AND MACHINE PARAMETERS
    # ============================

    intensity = intensity         # *bunch* intensity
    n_turns = 200          # simulation time
    n_macroparticles = int(n_mp) # number of macroparticles *per bunch*

    charge = e
    mass = m_p
    alpha = 0.0019236688211757462 #momentum compaction factor q20:18, q26:22.8 gamma_t=1/np.sqrt(alpha)
    gamma = 27.728550064942279
    p0 = np.sqrt(gamma**2 - 1.) * mass * c

    #Tune: Q26 optics nominal tunes
    accQ_x = 26.13 
    accQ_y = 26.18
    fracQ_y = int(10000*(np.modf(accQ_y)[0]))
    circumference = 6911.5038378975451

    # Optics
    s = None
    alpha_x = None
    alpha_y = None
    beta_x = 42 #q26:42, q20: 54.644808743169399
    beta_y = 42 #q26:42, q20: 54.509415262636274
    D_x = 0 #q26: 2.3, q20:3.75
    D_y = 0
    optics_mode = 'smooth'
    name = None
    n_segments = 1
    # apperture to calculate losses
    # when particles scape the aperture are accounted as losses
    apt_xy = aperture.EllipticalApertureXY(x_aper=0.07, y_aper=0.03)
    # self.machine.one_turn_map.append(apt_xy)

    # Implementation of octupoles
    KLOF=0
    KLOD=0
    Octupoles=SPSOctupolesNew(optics='Q26')
    app_x = 2 * p0 * Octupoles.get_anharmonicities(KLOF,KLOD)[0]
    app_xy= 2 * p0 * Octupoles.get_anharmonicities(KLOF,KLOD)[1]
    app_y = 2 * p0 * Octupoles.get_anharmonicities(KLOF,KLOD)[2]

    # Chromaticity and amplitude detuning from octupoles
    qph = 0.2 #knob
    qpv = qpv_val 
    Qp_x = np.array([qph*accQ_x, Octupoles.get_q2(KLOF,KLOD)[0]+500,-400000]) #q26 optics
    Qp_y = np.array([qpv*accQ_y, Octupoles.get_q2(KLOF,KLOD)[1]+130,+200000])

    # RF parameters (not really relevant here, since linear tracking)
    longitudinal_mode = 'non-linear'
    h_RF = [ 4620, 4*4620 ] #should be changed for q26?
    V_RF = [ 4.5e6, 0.45e6 ]
    dphi_RF = [ 0., np.pi ]
    wrap_z = False

    machine = Synchrotron(
            optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=name,
            alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), V_RF=V_RF, dphi_RF=dphi_RF, p0=p0, p_increment=0.,
            charge=charge, mass=mass, wrap_z=wrap_z)

    h = np.min(machine.longitudinal_map.harmonics) * 1

    # BEAM
    # ====
    epsn_x = 1.5e-6
    epsn_y = 1.5e-6
    sigma_z = 0.23
    bunch = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)	
    #bunch.y +=1e-3
    #bunch.x +=1e-3 #no kick?

    # CREATE BEAM SLICERS
    # ===================
    # IMPORTANT TO USE FIXED LONGITUDINAL CUTS!
    # Pass correct h_bunch, also considered above when defining filling scheme
    h_bunch = 924
    slicer_for_wakefields = UniformBinSlicer(n_slices, z_cuts=(-3.*sigma_z, 3.*sigma_z),
                                             circumference=circumference, h_bunch=h_bunch)	#it was 40 and 20 and h_bunch=h_bunch
    slicer_for_monitor = UniformBinSlicer(n_slices, z_cuts=(-4.*sigma_z, 4.*sigma_z),
                                          circumference=circumference, h_bunch=h_bunch)


    # WAKEFIELD
    # =========
    # Depending on number of bunches in beam: use 'memory_optimized', 'linear_mpi_full_ring_fft',
    # (or 'circular_mpi_full_ring_fft') for best performance (see slides).
    # Circular_fft not robust when using quad wakes, or when other elements in ring.
    # General advice: use either memory_optimized or linear_mpi_full_ring_fft.
    # n_turns_wake: memory for wake field. See slides for convention (has changed compared to HEADTAIL).
    
    n_turns_wake = 5  #number of turns the wakefield is alive
    wakefile1 = ('new_model_corr.wake')            # complete impedance model 2023
    #wakefile1 = ('Q26_wo_RS.wake')     # impedance model without resistive wall contribution
    ww1 = WakeTable(wakefile1, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], n_turns_wake=n_turns_wake)
    wake_field = WakeField(slicer_for_wakefields, ww1)
                            #mpi='linear_mpi_full_ring_fft', Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)

    # Adding a resonator Q=100, f=2.2GHz, R=2e6 Ohm/m                    
    r_h_dip = Resonator(R_shunt=2.e6, frequency=2.5e9, Q=100., Yokoya_X1=1., Yokoya_Y1=0., Yokoya_X2=0., Yokoya_Y2=0., switch_Z=False, n_turns_wake=n_turns_wake)
    r_h_quad = Resonator(R_shunt=2.6*1.0, frequency=2.5e9, Q=100., Yokoya_X1=0., Yokoya_Y1=0., Yokoya_X2=-1., Yokoya_Y2=0., switch_Z=False, n_turns_wake=n_turns_wake)
    r_v_dip = Resonator(R_shunt=2.e6, frequency=2.5e9, Q=100., Yokoya_X1=0., Yokoya_Y1=1., Yokoya_X2=0., Yokoya_Y2=0., switch_Z=False, n_turns_wake=n_turns_wake)
    r_v_quad = Resonator(R_shunt=2.e6*1.0, frequency=2.5e9, Q=100., Yokoya_X1=0., Yokoya_Y1=0., Yokoya_X2=0., Yokoya_Y2=1., switch_Z=False, n_turns_wake=n_turns_wake)

    wake_field_rhdip = WakeField(slicer_for_wakefields, r_h_dip)
    wake_field_rhquad = WakeField(slicer_for_wakefields, r_h_quad)
    wake_field_rvdip = WakeField(slicer_for_wakefields, r_v_dip)
    wake_field_rvquad = WakeField(slicer_for_wakefields, r_v_quad)

    machine.one_turn_map.append(wake_field)
    #machine.one_turn_map.append(wake_field_rhdip)
    #machine.one_turn_map.append(wake_field_rhquad)
    #machine.one_turn_map.append(wake_field_rvdip)
    #machine.one_turn_map.append(wake_field_rvquad)


    # TRANSVERSE FEEDBACK SYSTEM
    # ==========================
    dampingrate = 300 #Number of turns to damp the signal to 1/e. the lower the stronger
    damper = TransverseDamper(dampingrate_x=dampingrate, dampingrate_y=dampingrate, 
                              local_beta_function=beta_y)

    #machine.one_turn_map.append(damper)


    # MONITORS
    # ========
    # There is one bunchmonitor file for the whole beam.
    # f = h5py.File('bunchmonitor.h5')
    # f['Bunches'].keys() will give the keys for all the bunches that are stored in file.
    # To plot mean_x of "3rd bunch" for example: plt.plot(f['Bunches']['3']['mean_x'])

    activated_bunchmonitor = True
    n_batches = 1
    batch_length = 1
    bunchmonitor = BunchMonitor(
         f'./bunchmonitor_qpv{qpv}_N{intensity:.2e}_nonlinearq26', n_turns,
         simulation_parameters_dict={}, write_buffer_every=n_turns, buffer_size=n_turns)

    activated_slicemonitor = False
    
    # Slicemonitor cannot yet handle multi-bunch beams. Need to do manully with split_to_views()
    # as shown below. 1 file per bunch is created, but can also select smaller number of bunches.
    n_trns_slice_max = n_turns
    #slicemonitor = SliceMonitor(
    #     f'./slicemonitor_qpv{qpv}_nonlinearq26', n_trns_slice_max, slicer_for_monitor, write_buffer_every=n_turns, buffer_size=n_turns)
    
    activated_particlemonitor = False
    # Only save 1every tenth particle ... stride defines the stepsize (i.e. how many
   	# particles are skipped when saving to monitor)
    stride = 100
    # particlemonitor = ParticleMonitor(f'./particlemonitor_qpv{qpv}_nonlinearq26', stride=stride)

    # TRACKING LOOP
    # =============
    print('\n--> Begin tracking...\n')
    n_trns_slice = 0

    for i in tqdm(list(range(n_turns))):
        t0 = time.time()
        if i ==1:
            bunch.y +=1e-3
            bunch.x +=0
        machine.track(bunch)
        
        # Dump bunch and slice monitor data
        bunchmonitor.dump(bunch)

        # This is only for slice monitor 
        #if activated_slicemonitor:
        #    if activated_slicemonitor and (n_trns_slice < n_trns_slice_max):
        #        slicemonitor.dump(bunch)
        #        n_trns_slice += 1

        #if activated_particlemonitor:
        #        particlemonitor.dump(bunch)

        #print(('Turn: {:4d}, \tTime: {:3s}'.format(i, str(time.time() - t0))))
    print('\n*** Successfully completed!')


if __name__=="__main__":
    
    #study growth rate for negative chromaticity QPv 
    
    qpv_vect = np.linspace(0,1,10)

    intensity_vect=np.linspace(3e10,8e10,10)
    #n_macroparticles / n_slices should be > 2000
    n_slices_0 = 300 #de 50 parriba (hasta 200)
    n_mp_0 = 1e6 #era 3e5

    storage_path='./results'


    for qpv in qpv_vect:
        for intensity in intensity_vect:
                print('--------------------------------------------------')
                print(f"Running chromaticity sweep, qpv : {qpv}...")
                run(qpv_val=qpv, n_slices=n_slices_0, n_mp=n_mp_0, intensity=intensity)
                
                if not os.path.exists(storage_path):
                    os.makedirs(storage_path)
                for f in glob.glob('./*.h5'):
                    shutil.move(f, storage_path)
                for f in glob.glob('./*.h5part'):
                    shutil.move(f, storage_path)

    


