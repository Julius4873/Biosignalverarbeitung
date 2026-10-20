#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from scipy import fftpack
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
from scipy import fftpack
matplotlib.use('Qt5Agg')  # real GUI window for mouse interaction
import os
import re
import glob

""" This function does an FFT and returns the power spectrum of data
    Input: Data, sampling frequency
    Output: Power, the respective frequencies of the power spectrum
   """
def get_power(data, sfreq):
    sig_fft = fftpack.fft(data)
    
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)
    
    # The corresponding frequencies
    sample_freq1 = fftpack.fftfreq(data.size, d=1/sfreq)
    frequencies = sample_freq1[sample_freq1 > 0]
    power = power[sample_freq1 > 0]
    return power, frequencies

""""This function allows the user to select bursts in a signal using mouse clicks in a GUI window.
    Input: signal, time vector, number of bursts to select, title for the plot
    Output: start and end indices of the selected bursts
    Sample call: mvc_start, mvc_end = get_bursts(mvc_emg_envelope)
    Before calling the function change to "%matplotlib qt" in the notebook to enable GUI interaction, and after selection, 
    switch back to "%matplotlib inline" for inline plotting.
    """



def get_bursts(signal, time=None, number_bursts=3, title=''):
    """
    Interactive burst selection with the mouse, in a real GUI window.
    Click START then END for each burst, `number_bursts` times.

    Parameters
    ----------
    signal : array-like
        The signal to select bursts from.
    time : array-like, optional
        Time vector matching `signal`. If not given, sample indices
        (0, 1, 2, ...) are used as the x-axis instead.
    number_bursts : int
        How many start/end pairs to collect (default 3).
    title : str
        Optional label shown in the plot title.

    Returns
    -------
    starts, ends : np.ndarray of int
        Sample indices of the start and end of each burst.
    """
    signal = np.asarray(signal)
    if time is None:
        time = np.arange(len(signal))
    else:
        time = np.asarray(time)

    starts = []
    ends = []

    for b in range(1, number_bursts + 1):
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, color='blue')
        plt.title(f"{title}\nClick START and END for burst {b}/{number_bursts}")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.grid(True)

        pts = plt.ginput(2, timeout=-1)
        plt.close()

        if len(pts) != 2:
            raise ValueError("Did not select exactly 2 points!")

        start_x, end_x = pts[0][0], pts[1][0]
        if start_x > end_x:
            start_x, end_x = end_x, start_x

        start_idx = int(np.searchsorted(time, start_x))
        end_idx = int(np.searchsorted(time, end_x))

        print(f"Burst {b}: {start_x:.2f} - {end_x:.2f}  (indices {start_idx}-{end_idx})")

        starts.append(start_idx)
        ends.append(end_idx)

    return np.array(starts), np.array(ends)

"""This function imports the available EMG datasets from the current working directory.

The data files must be named ``MVC1.csv``, ``Weight2.csv`` or ``Fatigue3.csv``.
The loader  supports Arduino-style CSV files with 2 columns and headers named ``timestamp_us;adc``.

The function returns three variables in the order ``weights, mvc, fatigue``. 
Any missing group is ignored. 
If a file is missing inside a group, an error rises 
that includes information on which dataset is missing. That group is skipped.
"""

def import_data(separator=';'):
    """ This function is when you put together several datasets,
    but each dataset always starts with a time of 0.
    Input files are expected to have columns: timestamp_us, adc.
    Output: continuous time over all datasets,
    for each dataset type (Weight, MVC, Fatigue) independently.
    Each type requires files 1, 2, and 3 to all be present.
    """
    REQUIRED_INDICES = [1, 2, 3]

    def time_norm(data):
        a = list(data.iloc[:]['t'])
        b = list(data.iloc[:]['t'])
        offset = 0

        for u in range(len(a) - 1):
            if a[u] > a[u + 1]:
                if b[u] > b[u + 1]:
                    offset = a[u] - a[u + 1] + 1
                    a[u + 1] = offset + a[u + 1]
                else:
                    a[u + 1] = offset + a[u + 1]

        output = pd.DataFrame({'emg': data.emg, 't': a})
        output.reset_index(inplace=True, drop=True)
        return output

    def available_indices(prefix):
        """Find which numeric suffixes exist for a given file prefix,
        e.g. Weight1.csv, Weight2.csv -> [1, 2]"""
        indices = set()
        for path in glob.glob(f'{prefix}*.csv'):
            match = re.match(rf'^{prefix}(\d+)\.csv$', os.path.basename(path))
            if match:
                indices.add(int(match.group(1)))
        return indices

    def read_one_file(file_string):
        """Read a single raw file and convert to the emg/t format
        used downstream."""
        raw = pd.read_csv(
            file_string,
            sep=separator,
            header=0,
            skiprows=range(1, 51),
            skipfooter=50,
            engine='python'
        )

        # normalize column names: strip whitespace/BOM, lowercase
        raw.columns = [str(c).strip().lstrip('\ufeff').lower() for c in raw.columns]

        expected = {'timestamp_us', 'adc'}
        if not expected.issubset(raw.columns):
            raise ValueError(
                f"{file_string}: expected columns {expected}, "
                f"but found {list(raw.columns)}. Check the separator/header."
            )

        raw = raw.rename(columns={'timestamp_us': 't', 'adc': 'emg'})
        raw = raw[['emg', 't']]
        return raw

    def load_type(prefix):
        """Load files 1, 2, 3 for one dataset type, only if ALL three
        are present. Returns None and prints an error otherwise."""
        found = available_indices(prefix)
        missing = [i for i in REQUIRED_INDICES if i not in found]

        if missing:
            missing_files = [f'{prefix}{i}.csv' for i in missing]
            print(
                f"ERROR: {prefix} datasets are missing — cannot find "
                f"{', '.join(missing_files)}. Skipping {prefix}."
            )
            return None

        raw = pd.DataFrame(columns=['emg', 't'])
        for i in REQUIRED_INDICES:
            file_string = f'{prefix}{i}.csv'
            raw = pd.concat([raw, read_one_file(file_string)], ignore_index=True)

        return time_norm(raw)

    weights = load_type('Weight')
    mvc = load_type('MVC')
    fatigue = load_type('Fatigue')

    return weights, mvc, fatigue