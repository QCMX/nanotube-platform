# -*- coding: utf-8 -*-
"""
Identifies possible chiralities for a given list of peaks
in a fixed measurement window, based on Kataura predictions,
or experimental data from
https://doi.org/10.1038/nnano.2012.52

@author: sa
"""
import os
import numpy as np
import pandas as pd


# Load tables from CSV files
data_exp_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'References/Kaihui Liu.csv'), index_col=0)
data_kataura_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'References/Kataura.csv'), sep=';')


def read_exp_transitions(n,m):
    '''
    Reads the experimental values for a certain (n,m) chirality

    Parameters
    ----------
    n,m : integers
        NT chirality.

    Returns
    -------
    A list of transition energies in eV.
    '''
    p=n-m
    if n>=36:
        return []
    x = data_exp_df.at[n,str(p)]
    raw_list = x.replace('[','').replace(']','').split(' ')
    transitions = []
    for t in raw_list:
        try:
            transitions.append(float(t))
        except:
            pass
    return transitions


def read_kataura_transitions(n,m):
    '''
    Reads the theoretical values for a certain (n,m) chirality

    Parameters
    ----------
    n,m : integers
        NT chirality.

    Returns
    -------
    A list of transition energies in eV or
    an empty list if the chirality doesn't exist in the table.
    '''
    transitions = []
    df = data_kataura_df
    try:
        T = df.loc[(df.n==n) & (df.m==m)].iloc[:,-6:].values[0]
    except:
        return transitions

    for t in T:
        try:
            e = float(t.replace(',','.'))
            transitions.append(e)
        except:
            pass
    return transitions


def is_in_list(t, l, delta=0.05):
    '''
    Verifies that a transition exists in a list of peaks

    Parameters
    ----------
    t : float
        transition.
    l : list
        list of peaks.
    delta : float, optional
        Uncertainty of the position of the peak. The default is 0.02.

    Returns
    -------
    boolean
    '''
    for peak in l:
        if np.round(np.abs(t-peak), 3) <= delta:
            return True
    return False


def ntid2(peak1, peak2=None, d1=0.1, d2=0.1):
    '''
    Identifies possible chiralities for one or two peaks in table of
    theoretical transitions by Kataura.

    Parameters
    ----------
    peak1 : float
        Value of the first peak.
    peak2 : float, optional
        Value of the second peak
    d1 : float, optional
        Uncertainty of the position of the peak compared to the reference data. The default is 0.05.
    d2 : float, optional
        Uncertainty of the position of the peak compared to the reference data. The default is 0.05.

    Returns
    -------
    list of tuples with the format (n,m)
    '''
    candidates={}
    # candidates_sort={}
    a=0
    list_n, list_m = np.sort(data_kataura_df.n.unique()), np.sort(data_kataura_df.m.unique())
    for n in list_n:
        for m in list_m:
            for t in read_kataura_transitions(n,m):
                if np.round(np.abs(t-peak1),3)<=d1:
                    candidates[a] = (n,int(m))
            a+=1
    
    if peak2 is not None:
        finalists={}
        # finalists_sort={}
        a=0
        list_candidates = list(candidates.values())
        for candidate in list_candidates:
            (n,m)=candidate
            transitions = read_kataura_transitions(n, m)
            for t in transitions:
                if np.round(np.abs(t-peak2),3)<=d2:
                    finalists[a] = (n,m)
            a+=1            
        
        return list(finalists.values())
    return list(candidates.values())


def ntid_th(list_peaks, value_peaks, meas_window=[1.3,2.90], 
            delta=0.1, main_peak_factor=10):
    '''
    Identifies possible chiralities for a given list of peaks in a fixed measurement window, based on kataura predictions

    Parameters
    ----------
    list_peaks : list
        List of peaks detected in a spectrum, sorted by maximum value.
    value_peaks : list
        List of the intensities of the peaks, in the same order
    meas_window : list, optional
        Measurement window of the spectrum, in eV. The default is [1.25,3.00].
    delta : float, optional
        Uncertainty of the position of the peak compared to the reference data. The default is 0.05.
    main_peak_factor : float, optional
        Factor to discard small peaks compared to the main one. The default is 10.

    Returns
    -------
    list of tuples with the format (n,m)
    '''
    finalists=[]
    if len(list_peaks)==0:
        return finalists
    elif len(list_peaks)==1 or value_peaks[0]>main_peak_factor*value_peaks[1]:
        candidates = ntid2(list_peaks[0],d1=delta)
        for cd in candidates:
            a=0
            for transition in read_kataura_transitions(cd[0],cd[1]):
                if transition>=meas_window[0] and transition<=meas_window[1] and np.round(np.abs(transition-list_peaks[0]),3)>delta:
                    a=1
            if a==0:
                finalists.append(cd)
    elif len(list_peaks)>=2:
        candidates = ntid2(peak1=list_peaks[0], peak2=list_peaks[1],d1=delta,d2=delta)
        for cd in candidates:
            a=0
            for transition in read_kataura_transitions(cd[0],cd[1]):
                if transition>=meas_window[0] and transition<=meas_window[1] and not is_in_list(transition, list_peaks, delta=delta):
                    a=1
            if a==0:
                finalists.append(cd)
    return finalists


def ntid_exp(list_peaks, value_peaks, meas_window=[1.3,2.9], 
             delta_th=0.1, delta_exp=0.05, main_peak_factor=10):
    '''
    Identifies possible chiralities for a given list of peaks 
    in a fixed measurement window, based on kataura predictions, 
    then compares with the experimental data from 
    https://doi.org/10.1038/nnano.2012.52

    Parameters
    ----------
    list_peaks : list
        List of peaks detected in a spectrum, sorted by maximum value.
    value_peaks : list
        List of the intensities of the peaks, in the same order
    meas_window : list, optional
        Measurement window of the spectrum, in eV. The default is [1.3,2.9].
    delta_th : float, optional
        Uncertainty on the position of the peak compared to the theoretical
        predictions from the kataura plot. The default is 0.1.
    delta_exp : float, optional
        Uncertainty on the position of the peak compared to the experimental
        data. The default is 0.05.
    main_peak_factor : float, optional
        Factor to discard small peaks compared to the main one. The default is 10.

    Returns
    -------
    finalists : list of tuples with the format (n,m).

    '''
   
    candidates = ntid_th(list_peaks, value_peaks, 
                           meas_window=meas_window, 
                           delta=delta_th, main_peak_factor=main_peak_factor)
    finalists=[]
       
    for cd in candidates:
        a=0
        list_transitions_exp = list(dict.fromkeys(read_exp_transitions(cd[0], cd[1])))
        for transition in list_transitions_exp:
            if transition<=meas_window[0] or transition>=meas_window[1] or is_in_list(transition, list_peaks, delta=delta_exp):
                a+=1
        if a==len(list_transitions_exp):
            finalists.append(cd)
    return finalists


def string_chirality(n,m):
    if (n-m)%3==0:
        return('M({},{}); '.format(n,m))
    else:
        return('S({},{}); '.format(n,m))
