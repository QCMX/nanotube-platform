# -*- coding: utf-8 -*-
"""
Identify carbon nanotube chiralities by observed optical transitions.
Based on data by Kataura and Kaihui Liu et al.
"""
import os
import functools
import numpy as np
import pandas as pd


# Put both datasets into same format
# Dataframe with columns: n, m, transition1, transition2, transition3, ...
# transitions possibly nan
def load_Kataura():
    """Load data for Kataura plot from local file.

    Needs __file__ variable for relative path, so doesn't work when calling
    directly in interactive Python.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'References/Kataura.csv'), sep=';')
    df = df[['n', 'm']+list(df.columns[6:])]
    df.iloc[:,2:] = df.iloc[:,2:].map(lambda s: float(s.replace(',', '.')) if type(s)==str else s)
    df = df.infer_objects()
    df = df.sort_values('m').sort_values('n').reset_index(drop=True)
    return df

def load_KaihuiLiu():
    """Load experimental CNT atlas from local file.

    Needs __file__ variable for relative path, so doesn't work when calling
    directly in interactive Python.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'References/Kaihui Liu.csv'), index_col=0)
    df.columns = pd.to_numeric(df.columns)
    data = []
    for n, row in df.iterrows():
        for nm, cell in row.items():
            m = n - nm
            if m < 0:
                break
            assert cell.startswith('[') and cell.endswith(']')
            transitions = [
                float(s) for s in cell.replace('[', '').replace(']', '').split(' ')
                if s]
            if len(transitions) == 0:
                continue
            data.append([n,m]+transitions)
    maxntrans = max(len(l) for l in data) - 2
    coln = ['n', 'm'] + ['transition%d'%d for d in range(maxntrans)]
    return pd.DataFrame(data, columns=coln)


# Load tables from CSV files
data_Kataura = load_Kataura()
data_KaihuiLiu = load_KaihuiLiu()


def select_chiralities_matching_any(energies, delta, dataset):
    """Select rows of dataset that contain one or more of the
    given transitions energies.

    Returns view of dataset.
    """
    x = [(np.abs(dataset.iloc[:,2:] - energy) <= delta).any(axis=1) for energy in energies]
    x = functools.reduce(lambda a, b: a | b, x)
    return dataset.loc[x]


def select_chiralities_matching_all(energies, delta, window, dataset):
    """Select rows of dataset where all tabulated transitions in the
    window are also listed in observed `energies`.

    Parameters
    ----------
    energies : list of float
        List of observed transition energies.
    delta : float
        Maximum deviation from tabulated transitions.
    window : 2-tuple of float
        Observation window.
    dataset : TYPE
        DataFrame of tabulated transitions.

    Returns
    -------
    DataFrame
        view of matching transitions.
    """
    # True for transitions in window
    inwindow = (dataset.iloc[:,2:] > window[0]) & (dataset.iloc[:,2:] < window[1])
    # True for rows with no transitions in window
    alloutofwindow = (~inwindow).all(axis=1)
    # For each energy: True for transitions close to energy or out of window
    x = [(np.abs(dataset.iloc[:,2:] - energy) <= delta) | ~inwindow for energy in energies]
    # True for transitions close to any energy or out of window
    x = functools.reduce(lambda a, b: a | b, x)
    # True for rows with all transitions close to one energy or out of window
    x = x.all(axis=1) & (~alloutofwindow)
    return dataset.loc[x]


def transitions_in_other_table(first, second):
    """Select rows of second table with (n,m) appearing in first table.

    Does not complain if (n,m) of first table doesn't appear in second one.

    Returns view of `second` DataFrame.
    """
    # Using multiindex: faster and conserves order of first table
    return first[['n','m']].set_index(['n', 'm']).join(second.set_index(['n', 'm'])).reset_index()
    # # Using masks:
    # x = [(second['n']==row['n']) & (second['m']==row['m']) for _, row in first.iterrows()]
    # x = functools.reduce(lambda a, b: a | b, x)
    # return second[x]


def get_tabulated_th(dfnm):
    return transitions_in_other_table(dfnm, data_Kataura)


def get_tabulated_exp(dfnm):
    return transitions_in_other_table(dfnm, data_KaihuiLiu)


def nt_id(energies, intensities, delta_th, delta_exp, window, main_peak_factor):
    """Composite nanotube identification:

    Disregard peaks smaller than main_peak_factor compared to first peak.
    Select expected theoretical chiralities (Kataura) where all transitions in the
    measurement window are also in the observed peak energies.
    Of these select only those where no other transitions are expected
    expected based on Kaihui Liu et al.

    Changed to Samy's original algorithm: take more than at most two peaks into account.

    Sort by first unmatched peak of decreasing intensity.
    At lower priority sort by closeness to matched peaks (compared to Kaihui Liu et al.).

    Parameters
    ----------
    energies : list of floats
        Observed peak positions.
    intensities : list of floats
        Observed peak intensities.
    delta_th : float
        Maximum deviation from tabulated theoretical transitions (Kataura).
    delta_exp : float
        Maximum deviation from tabulated experimental transitions (Kaihui Liu).
    window : 2-tuple of floats
        Lower and upper bound of measurement window
    main_peak_factor : TYPE
        Disregards peaks smaller, relative to main peak, than this factor.

    Returns
    -------
    DataFrame with columns n, m.
    """
    energies, intensities = np.array(energies), np.array(intensities)
    # Sort by intensity, descending
    s = np.argsort(intensities)[::-1]
    energies, intensities = energies[s], intensities[s]

    # Discard small peaks
    mask = (intensities[0] / intensities) < main_peak_factor
    energies, intensities = energies[mask], intensities[mask]

    candidates = select_chiralities_matching_all(energies, delta_th, window, data_Kataura)
    candidates = transitions_in_other_table(candidates, data_KaihuiLiu)
    candidates = select_chiralities_matching_all(energies, delta_exp, window, candidates)

    # Sort results
    def sortbadness(row):
        ts = row.iloc[2:]
        unmatchedbadness, dist = 0, []
        for i, energy in enumerate(energies):
            d = np.min(np.abs(ts - energy))
            if d <= delta_exp:
                dist.append(d)
            elif unmatchedbadness == 0:
                unmatchedbadness = len(energies)-i
        return np.sum(dist)/len(dist) + unmatchedbadness*1e6
    candidates['sortbadness'] = candidates.apply(sortbadness, axis=1)
    candidates = candidates.sort_values('sortbadness')

    return candidates[['n', 'm']]
