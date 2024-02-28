# -*- coding: utf-8 -*-
"""
Notebook like script (for editors understanding #%% as cell separator)
for plotting NT spectra.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import constants as si
from scipy.signal import find_peaks

import nt_identification as ntid

#%%
# Load laser spectrum for normalization

LASER_CALIBRATION_FILES = '../220617 laser calibration/220622/data_laser/'
laser_power = 45 # percent
IR_att = 40 # dB
background_intensity = 550

# Search for calibration corresponding to laser settings
fnamematches = [
    filename for filename in os.listdir(LASER_CALIBRATION_FILES)
    if filename.startswith(f'P{laser_power:d}_{IR_att:d}')]
assert len(fnamematches) == 1, "Need to match exactly one calibration file"
calibrationpath = os.path.join(LASER_CALIBRATION_FILES, fnamematches[0])

# Load spectrum
data_laser = np.genfromtxt(calibrationpath, delimiter=',').T
lambdas_laser = data_laser[0]
intensities_laser = data_laser[1] - background_intensity

#%%

def string_chirality(n,m):
    """Make string describing chirality."""
    if (n-m)%3 == 0:
        return 'M({},{})'.format(n,m)
    else:
        return 'S({},{})'.format(n,m)

def plot_overview(
        filepath,
        roi_background_rows = slice(0, 60),
        roi_nt_rows=slice(320, 380),
        meas_window = (1.3, 2.7), # eV
        peak_width=20, # samples
        main_peak_factor=5,
        delta_exp=0.05, delta_th=0.1,
        savepath=None):
    filename = os.path.basename(filepath)
    filedata = np.genfromtxt(filepath, delimiter=',')
    assert filedata.shape[0] % 1340 == 0

    lambdas = filedata[:1340,0] # nm
    energies = si.h*si.c/(lambdas*1.e-9*si.e) # eV
    counts = filedata[:,1].reshape((-1, 1340))
    assert np.allclose(lambdas, lambdas_laser), "Data wavelengths needs to match laser ref"

    bkg = np.sum(counts[roi_background_rows], axis=0)
    spectrum = np.sum(counts[roi_nt_rows], axis=0) - bkg
    normalized = spectrum / intensities_laser

    # Find peaks & identify
    peaks_ind = find_peaks(normalized, width=peak_width)[0]
    xpeaks, ypeaks = energies[peaks_ind], normalized[peaks_ind]

    order = np.argsort(ypeaks)[::-1]
    id_peaks = ntid.nt_id(
        energies=xpeaks[order],
        intensities=ypeaks[order],
        delta_th=delta_th, delta_exp=delta_exp,
        window=meas_window,
        main_peak_factor=main_peak_factor)

    # Plot
    fig = plt.figure(layout='constrained', figsize=(10, 6))
    axd = fig.subplot_mosaic(
        """
        AC
        BD
        """)
    axd['B'].set_title('Full sensor data')
    im = axd['B'].imshow(counts, origin='upper', aspect='auto', interpolation='nearest')
    axd['B'].set_xlabel('detector column')
    axd['B'].set_ylabel('detector row')
    axd['B'].axhline(roi_background_rows.start, color='C1', linestyle='--', linewidth=1, label='ROI background')
    axd['B'].axhline(roi_background_rows.stop, color='C1', linestyle='--', linewidth=1)
    axd['B'].axhline(roi_nt_rows.start, color='r', linestyle='--', linewidth=1, label='ROI spectrum')
    axd['B'].axhline(roi_nt_rows.stop, color='r', linestyle='--', linewidth=1)
    axd['B'].legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=axd['B'], orientation='horizontal', shrink=0.8).set_label('Detector counts', fontsize=8)

    axd['A'].set_title('Raw spectrum')
    axd['A'].plot(lambdas_laser, intensities_laser/1e3, color='gray', label='laser reference')
    axd['A'].plot(lambdas, spectrum/1e3, label='nanotube')
    axd['A'].legend(fontsize=8)
    axd['A'].set_xlabel('Wavelength / nm')
    axd['A'].set_ylabel('Detector counts / 1k')

    axd['C'].plot(energies, normalized, linewidth=1)
    axd['C'].set_ylim(0, np.max(normalized[np.argmin(np.abs(energies-meas_window[1])):]*1.1))
    for i, (xpeak, ypeak) in enumerate(zip(xpeaks[order], ypeaks[order])):
        c = 'r' if ypeaks[order][i]*main_peak_factor > ypeaks[order][0] else 'gray'
        axd['C'].plot([xpeak, xpeak], [0, ypeak], '--', color=c, label=f'{xpeak:.2f} eV')
    axd['C'].legend(loc='upper right', fontsize=8)
    axd['C'].set_xlabel('Transition energy / eV')
    axd['C'].set_ylabel('Normalized counts')
    axd['C'].set_title('Normalized spectrum')

    TRUNCATE = 20
    id_peaks_trunc = id_peaks.iloc[:TRUNCATE]
    axd['D'].set_title('Candidate signatures'+(f' (truncated, {TRUNCATE} of {len(id_peaks)})' if len(id_peaks) > TRUNCATE else ''))
    axd['D'].sharex(axd['C'])
    for i, (xpeak, ypeak) in enumerate(zip(xpeaks[order], ypeaks[order])):
        c = 'r' if ypeaks[order][i]*main_peak_factor > ypeaks[order][0] else 'gray'
        axd['D'].axvline(xpeak, linewidth=0.8, color=c, zorder=-1)
    for j, (_, row) in enumerate(ntid.get_tabulated_exp(id_peaks_trunc).iterrows()):
        for e in row.iloc[2:]:
            if e > max(energies) or e < min(energies): continue
            axd['D'].plot([e, e], [j, j+1], color='r', linewidth=3)
    for j, (_, row) in enumerate(ntid.get_tabulated_th(id_peaks_trunc).iterrows()):
        for e in row.iloc[2:]:
            if e > max(energies) or e < min(energies): continue
            axd['D'].plot([e, e], [j, j+1], color='k', linewidth=1)
    axd['D'].set_yticks(
        np.arange(len(id_peaks_trunc))+0.5,
        [string_chirality(n,m) for _, (n,m) in id_peaks_trunc.iterrows()],
        fontsize=8)
    axd['D'].set_ylim([len(id_peaks_trunc), 0])

    fig.suptitle(filename+'\n Laser calibration: '+os.path.basename(calibrationpath), fontsize=8)

    if savepath is not None:
        fig.savefig(savepath)

#%%
# Make plot for a single file

plot_overview('../Growth/20230928_G55/P45_2s_G55_B79_NT7_2d962 2023-09-29 11_40_00.csv')

#%%
# Make plot for all CSV in the folder and save in subfolder 'overview/'

folder = '../Growth/20240112_G57'
if not os.path.exists(os.path.join(folder, 'overview')):
    os.mkdir(os.path.join(folder, 'overview'))
files = []
for fname in os.listdir(folder):
    if fname.endswith('.csv') and 'Roi' not in fname:
        print(fname)
        savepath = os.path.join(folder, 'overview', fname+'.png')
        plot_overview(os.path.join(folder, fname), savepath=savepath)
        plt.pause(0.1)
