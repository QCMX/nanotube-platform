# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:26:41 2022

@author: sa
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import constants as si
from scipy.signal import find_peaks

os.chdir('C:/Users/sa/Documents/Data/NT spectra')

import NT_identification as ntid

def candidates(list_peaks, delta_exp=0.05, meas_window=[1.3,2.9]):
    for ch in ntid.ntid_exp(list_peaks, [1,1,1], delta_exp=delta_exp, meas_window=meas_window):
        n,m=ch[0],ch[1]
        print(ntid.string_chirality(ch[0],ch[1]) + ' : EXP (' + str(ntid.read_exp_transitions(n, m)) + ') ; KAT (' + str(ntid.read_kataura_transitions(n, m)) + ')')
        # print(ntid.read_exp_transitions(n, m))      
        # print(ntid.read_kataura_transitions(n, m))
    return

'''load laser spectrum for normalization'''

laser_power = 45
IR_att = 40

background_intensity = 550

data_laser_path = 'C:/Users/sa/Documents/Data/NT spectra/Laser calibration/220617 laser calibration/220617 laser calibration/220622/data_laser/'
prefixed = [filename for filename in os.listdir(data_laser_path) if filename.startswith('P{p}_{ir}'.format(p=laser_power, ir=IR_att))]
data_file_laser = data_laser_path + prefixed[0]
data_laser = np.genfromtxt(data_file_laser, delimiter=',').T

lambdas_laser = data_laser[0]
intensities_laser = data_laser[1] - background_intensity

#%%
'''plot single file from full sensor data'''

data_NT_path = 'Growth/20231103_G56/P45_4s_G56_B86_NT12_1d506 2023-11-03 17_48_40.csv'
filename = data_NT_path.split('/')[-1].split('.')[0]
data_NT = np.genfromtxt(data_NT_path, delimiter=',').T

plot_full = False
identify = True
savefigure = False
normalize = False

'''integration region definition'''
width_integration = 60
start_roi_back, start_roi_NT = 0, 320
ROI_back = (start_roi_back, start_roi_back + width_integration)
ROI_NT = (start_roi_NT, start_roi_NT + width_integration)

'''identifying parameters definition'''
peak_width = 20
delta_th, delta_exp = 0.1, 0.05
main_peak_factor = 10

lambdas_NT = data_NT[0][0:1340]

if normalize:
    assert np.abs(lambdas_NT[0] - lambdas_laser[0]) <= 1., "Laser spectrum window doesn\'t match the data spectrum window"

energies = si.h*si.c/(lambdas_NT*1.e-9*si.e)
intensities = np.array([data_NT[1][1340*i:1340*(i+1)] for i in range(len(data_NT[0])//1340)])

if plot_full:
    fig,ax = plt.subplots(figsize=(10,5))
    ax.imshow(intensities)
    plt.title('Full spectrum, not normalized')
    plt.tight_layout()

else:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_xlabel('Energy [eV]')
    fig.suptitle(filename)
    d = np.sum(intensities[ROI_NT[0]:ROI_NT[1],:],axis=0) - np.sum(intensities[ROI_back[0]:ROI_back[1],:],axis=0)
        
    if not identify:   
        if normalize:
            ax.set_ylabel('Normalized intensity [a.u.]')
            ax.plot(energies, d/intensities_laser, ms=3., c='k')
        else:
            ax.set_ylabel('Intensity [counts]')
            ax.plot(energies, d, ms=3., c='k')
        plt.tight_layout()    
        
    else:
        if normalize:
            d = d/intensities_laser
            
        peaks_ind = find_peaks(d, width=peak_width)[0]
        xpeaks, ypeaks = energies[peaks_ind], d[peaks_ind]
        order = np.argsort(ypeaks)[::-1]
        xpeaks_round = np.round(xpeaks,2)
        id_peaks = ntid.ntid_exp(list_peaks=xpeaks_round[order], value_peaks = ypeaks[order], meas_window=[np.min(energies)+0.1,np.max(energies)-0.1], 
                            delta_th=delta_th, delta_exp=delta_exp,main_peak_factor=main_peak_factor)
        
        for xpeak, ypeak in zip(xpeaks[order], ypeaks[order]):
            ax.plot([xpeak, xpeak], [0, ypeak], 'r--', label='{:.2f} {}'.format(xpeak, 'eV'))
            leg = ax.legend(title='Peaks positions:', handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)
                
        text_chiralities=''
        for ch in id_peaks:
            text_chiralities+=ntid.string_chirality(ch[0],ch[1])
            
        ax.set_title('Possible Chiralities: ' + text_chiralities)
        if normalize:
            ax.set_ylabel('Normalized intensity [a.u.]')
        else:
            ax.set_ylabel('Intensity [counts]')
        ax.plot(energies, d, ms=3., c='k')            
        plt.tight_layout()
    
    if savefigure:
        if normalize:
            plt.savefig('/'.join(data_NT_path.split('/')[:-1]) + '/' + filename + '_N.png')
        else:
            plt.savefig('/'.join(data_NT_path.split('/')[:-1]) + '/' + filename + '.png')
        

#%%

'''plot all files from full sensor data'''

plot_full = False
identify = True
savefigure = True

files = []
folder = 'Growth/20231103_G56'
for (dirpath, dirnames, filenames) in os.walk(folder):
    files.extend(filenames)
    break

for f in files:
    if '.csv' in f and 'Roi' not in f:
        data_NT_path = folder + '/' + f
        filename = data_NT_path.split('/')[-1].split('.')[0]
        data_NT = np.genfromtxt(data_NT_path, delimiter=',').T

        '''integration region definition'''
        width_integration = 60
        start_roi_back, start_roi_NT = 0, 320
        ROI_back = (start_roi_back, start_roi_back + width_integration)
        ROI_NT = (start_roi_NT, start_roi_NT + width_integration)

        '''identifying parameters definition'''
        peak_width = 20
        delta_th, delta_exp = 0.1, 0.02
        main_peak_factor = 10
        
        normalize = True
        lambdas_NT = data_NT[0][0:1340]

        energies = si.h*si.c/(lambdas_NT*1.e-9*si.e)
        intensities = np.array([data_NT[1][1340*i:1340*(i+1)] for i in range(len(data_NT[0])//1340)])

        if plot_full:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.imshow(intensities)
            plt.title('Full spectrum, not normalized')
            plt.tight_layout()

        else:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.set_xlabel('Energy [eV]')
            fig.suptitle(filename)
            d = np.sum(intensities[ROI_NT[0]:ROI_NT[1],:],axis=0) - np.sum(intensities[ROI_back[0]:ROI_back[1],:],axis=0)
                
            if not identify:   
                ax.set_ylabel('Intensity [counts]')
                ax.plot(energies, d, ms=3., c='k')
                plt.tight_layout()    
                
            else:                    
                peaks_ind = find_peaks(d, width=peak_width)[0]
                xpeaks, ypeaks = energies[peaks_ind], d[peaks_ind]
                order = np.argsort(ypeaks)[::-1]
                xpeaks_round = np.round(xpeaks,2)
                id_peaks = ntid.ntid_exp(list_peaks=xpeaks_round[order], value_peaks = ypeaks[order], meas_window=[np.min(energies)+0.1,np.max(energies)-0.1], 
                                    delta_th=delta_th, delta_exp=delta_exp,main_peak_factor=main_peak_factor)
                
                for xpeak, ypeak in zip(xpeaks[order], ypeaks[order]):
                    ax.plot([xpeak, xpeak], [0, ypeak], 'r--', label='{:.2f} {}'.format(xpeak, 'eV'))
                    leg = ax.legend(title='Peaks positions:', handlelength=0, handletextpad=0, fancybox=True)
                    for item in leg.legendHandles:
                        item.set_visible(False)
                        
                text_chiralities=''
                for ch in id_peaks:
                    text_chiralities+=ntid.string_chirality(ch[0],ch[1])
                    
                ax.set_title('Possible Chiralities: ' + text_chiralities)
                ax.set_ylabel('Intensity [counts]')
                ax.plot(energies, d, ms=3., c='k')            
                plt.tight_layout()
            
            if savefigure:
                # plt.savefig('/'.join(data_NT_path.split('/')[:-1]) + '/' + filename + '.png')
                folder_raw = '/'.join(data_NT_path.split('/')[:-1]) + '/Raw spectra/'
                if 'Raw spectra' not in os.listdir('/'.join(data_NT_path.split('/')[:-1])):
                    os.mkdir(folder_raw)
                plt.savefig(folder_raw + filename + '.png')
        plt.close()
        plt.pause(0.1)

        
        assert lambdas_NT[0]==lambdas_laser[0], 'Laser spectrum window doesn\'t match the data spectrum window'

        energies = si.h*si.c/(lambdas_NT*1.e-9*si.e)
        intensities = np.array([data_NT[1][1340*i:1340*(i+1)] for i in range(len(data_NT[0])//1340)])

        if plot_full:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.imshow(intensities)
            plt.title('Full spectrum, not normalized')
            plt.tight_layout()

        else:
            fig,ax = plt.subplots(figsize=(10,5))
            ax.set_xlabel('Energy [eV]')
            fig.suptitle(filename)
            d = np.sum(intensities[ROI_NT[0]:ROI_NT[1],:],axis=0) - np.sum(intensities[ROI_back[0]:ROI_back[1],:],axis=0)
                
            if not identify:   
                ax.set_ylabel('Normalized intensity [a.u.]')
                ax.plot(energies, d/intensities_laser, ms=3., c='k')
                plt.tight_layout()    
                
            else:
                d = d/intensities_laser
                    
                peaks_ind = find_peaks(d, width=peak_width)[0]
                xpeaks, ypeaks = energies[peaks_ind], d[peaks_ind]
                order = np.argsort(ypeaks)[::-1]
                xpeaks_round = np.round(xpeaks,2)
                id_peaks = ntid.ntid_exp(list_peaks=xpeaks_round[order], value_peaks = ypeaks[order], meas_window=[np.min(energies)+0.1,np.max(energies)-0.1], 
                                    delta_th=delta_th, delta_exp=delta_exp,main_peak_factor=main_peak_factor)
                
                for xpeak, ypeak in zip(xpeaks[order], ypeaks[order]):
                    ax.plot([xpeak, xpeak], [0, ypeak], 'r--', label='{:.2f} {}'.format(xpeak, 'eV'))
                    leg = ax.legend(title='Peaks positions:', handlelength=0, handletextpad=0, fancybox=True)
                    for item in leg.legendHandles:
                        item.set_visible(False)
                        
                text_chiralities=''
                for ch in id_peaks:
                    text_chiralities+=ntid.string_chirality(ch[0],ch[1])
                    
                ax.set_title('Possible Chiralities: ' + text_chiralities)
                ax.set_ylabel('Normalized intensity [a.u.]')
                ax.plot(energies, d, ms=3., c='k')            
                plt.tight_layout()
            
            if savefigure:
                # print('Saving ' + '/'.join(data_NT_path.split('/')[:-1]) + '/' + filename + '_N.png')
                folder_normalized = '/'.join(data_NT_path.split('/')[:-1]) + '/Normalized spectra/'
                if 'Normalized spectra' not in os.listdir('/'.join(data_NT_path.split('/')[:-1])):
                    os.mkdir(folder_normalized)
                plt.savefig(folder_normalized + filename + '_N.png')

        plt.close()
        plt.pause(0.1)              
        
        
#%%
'''plot single file from ROI data'''

data_NT_path = 'Stock/20221024_stock/20221024_stock/G042_CMA51_NT1 2022-10-24 16_33_42-Roi-2.csv'
data_back_path = 'Stock/20221024_stock/20221024_stock/G042_CMA50_NT5 2022-10-24 16_27_04-Roi-3.csv'
filename = data_NT_path.split('/')[-1].split('.')[0]
filename_back = data_back_path.split('/')[-1].split('.')[0]
data_NT = np.genfromtxt(data_NT_path, delimiter=',').T
data_back = np.genfromtxt(data_back_path, delimiter=',').T

normalize = False
identify = False
savefigure = False


'''identifying parameters definition'''
peak_width = 10
delta_th, delta_exp = 0.1, 0.02
main_peak_factor = 10

lambdas_NT = data_NT[0]

if normalize:
    assert lambdas_NT[0]==lambdas_laser[0], 'Laser spectrum window doesn\'t match the data spectrum window'

energies = si.h*si.c/(lambdas_NT*1.e-9*si.e)
intensities = np.array(data_NT[1])
intensities_back = np.array(data_back[1])


fig,ax = plt.subplots(figsize=(10,5))
ax.set_xlabel('Energy [eV]')
fig.suptitle(filename)
d = intensities - intensities_back
    
if not identify:   
    if normalize:
        ax.set_ylabel('Normalized intensity [a.u.]')
        ax.plot(energies, d/intensities_laser, '.', ms=3., c='k')
    else:
        ax.set_ylabel('Intensity [counts]')
        ax.plot(energies, d, '.', ms=3., c='k')
    plt.tight_layout()    
    
else:
    if normalize:
        d = d/intensities_laser
        ax.set_ylabel('Normalized intensity [a.u.]')
    else:
        ax.set_ylabel('Intensity [a.u.]')
        
    peaks_ind = find_peaks(d, width=peak_width)[0]
    xpeaks, ypeaks = energies[peaks_ind], d[peaks_ind]
    order = np.argsort(ypeaks)[::-1]
    xpeaks_round = np.round(xpeaks,2)
    id_peaks = ntid.ntid_exp(list_peaks=xpeaks_round[order], value_peaks = ypeaks[order], meas_window=[np.min(energies)+0.1,np.max(energies)-0.1], 
                        delta_th=delta_th, delta_exp=delta_exp,main_peak_factor=main_peak_factor)
    
    for xpeak, ypeak in zip(xpeaks[order], ypeaks[order]):
        ax.plot([xpeak, xpeak], [0, ypeak], 'r--', label='{:.2f} {}'.format(xpeak, 'eV'))
        leg = ax.legend(title='Peaks positions:', handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)
            
    text_chiralities=''
    for ch in id_peaks:
        text_chiralities+=ntid.string_chirality(ch[0],ch[1])
        
    ax.set_title('Possible Chiralities: ' + text_chiralities)
    ax.plot(energies, d, '.', ms=3., c='k')            
    plt.tight_layout()

if savefigure:
    plt.savefig('/'.join(data_NT_path.split('/')[:-1]) + '/' + filename + '.png')

#%%
'''nice plot single file from full sensor data for publication'''

data_NT_path = 'C:/Users/sa/Documents/Data/NT spectra/Transfer/20230206_transfer/20230206_transfer/G050_A64_P45 2023-02-06 13_15_35.csv'
# data_NT_path = 'Stock/20221024_stock/20221024_stock/G042_CMB42_NT8 2022-10-24 16_43_14.csv'
filename = data_NT_path.split('/')[-1].split('.')[0]
data_NT = np.genfromtxt(data_NT_path, delimiter=',').T

normalize = False
plot_full = False
identify = False
savefigure = False

'''display parameters'''
tick_size = 18
axis_size= 25
# label='(n,m)=(12,6)'

'''integration region definition'''
width_integration = 60
start_roi_back, start_roi_NT = 100, 300
ROI_back = (start_roi_back, start_roi_back + width_integration)
ROI_NT = (start_roi_NT, start_roi_NT + width_integration)

'''identifying parameters definition'''
peak_width = 20
delta_th, delta_exp = 0.1, 0.02
main_peak_factor = 10

lambdas_NT = data_NT[0][0:1340]

if normalize:
    assert lambdas_NT[0]==lambdas_laser[0], 'Laser spectrum window doesn\'t match the data spectrum window'

energies = si.h*si.c/(lambdas_NT*1.e-9*si.e)
intensities = np.array([data_NT[1][1340*i:1340*(i+1)] for i in range(len(data_NT[0])//1340)])

if plot_full:
    fig,ax = plt.subplots(figsize=(12,8))
    ax.imshow(intensities)
    plt.title('Full spectrum, not normalized')
    plt.tight_layout()

else:
    fig,ax = plt.subplots(figsize=(12,8))
    ax.set_xlabel('Energy [eV]', fontsize=axis_size)
    d = np.sum(intensities[ROI_NT[0]:ROI_NT[1],:],axis=0) - np.sum(intensities[ROI_back[0]:ROI_back[1],:],axis=0)
        
    if not identify:   
        if normalize:
            ax.set_ylabel('Normalized intensity [a.u.]', fontsize=axis_size)
            ax.plot(energies, d/intensities_laser, ms=3., c='k')
        else:
            ax.set_ylabel('Intensity [counts]', fontsize=axis_size)
            ax.plot(energies, d, ms=3., c='k')
            
        # ax.legend(frameon=False, title=label, title_fontsize=axis_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.tight_layout()    
        
    else:
        if normalize:
            d = d/intensities_laser
            
        peaks_ind = find_peaks(d, width=peak_width)[0]
        xpeaks, ypeaks = energies[peaks_ind], d[peaks_ind]
        order = np.argsort(ypeaks)[::-1]
        xpeaks_round = np.round(xpeaks,2)
        id_peaks = ntid.ntid_exp(list_peaks=xpeaks_round[order], value_peaks = ypeaks[order], meas_window=[np.min(energies)+0.1,np.max(energies)-0.1], 
                            delta_th=delta_th, delta_exp=delta_exp,main_peak_factor=main_peak_factor)
        
        for xpeak, ypeak in zip(xpeaks[order], ypeaks[order]):
            ax.plot([xpeak, xpeak], [0, ypeak], 'r--', label='{:.2f} {}'.format(xpeak, 'eV'))
            leg = ax.legend(title='Peaks positions:', handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)
                
        text_chiralities=''
        for ch in id_peaks:
            text_chiralities+=ntid.string_chirality(ch[0],ch[1])
            
        ax.set_title('Possible Chiralities: ' + text_chiralities)
        ax.set_ylabel('Normalized intensity [a.u.]', fontsize=axis_size)
        ax.plot(energies, d, ms=3., c='k') 
        ax.legend(frameon=False)
        plt.tick_params(axis='both', which='major', labelsize=tick_size)           
        plt.tight_layout()
    
    if savefigure:
        plt.savefig('/'.join(data_NT_path.split('/')[:-1]) + '/' + filename + '.png')


#%%



    
    