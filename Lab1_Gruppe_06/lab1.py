"""Auswertung Labor 1. Keine Messwerte werden erzeugt oder stillschweigend verworfen."""
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

AXES = ('X', 'Y', 'Z')
G0 = 9.80665

@dataclass
class Record:
    name: str
    t_ms: np.ndarray
    raw: np.ndarray
    full_scale_g: float
    unit: str = 'counts'

    @property
    def t(self):
        return (self.t_ms - self.t_ms[0]) / 1000.0

    @property
    def g(self):
        return self.raw if self.unit == 'g' else self.raw / (2048.0 / self.full_scale_g)

    @property
    def ms2(self):
        return self.g * G0

def load_record(path, columns, full_scale_g):
    """Spaltenreihenfolge und Messbereich sind explizit anzugeben, nicht zu erraten.

    Erwartet 12-bit-Rohwerte (Counts), keine bereits umgerechneten mg/g-Werte.
    Leerzeilen sind erlaubt; beschädigte Zeilen, Zeitresets und Duplikate nicht.
    """
    path = Path(path)
    data = np.loadtxt(path, dtype=float, ndmin=2)
    if data.shape[1] != 4 or set(columns) != {'t_ms', 'X', 'Y', 'Z'}:
        raise ValueError(f'{path.name}: genau vier eindeutig benannte Spalten erwartet.')
    if full_scale_g not in (2, 4, 8) or not np.isfinite(data).all():
        raise ValueError(f'{path.name}: ungueltiger Messbereich oder nicht endliche Daten.')
    t_ms = data[:, columns.index('t_ms')]
    raw = data[:, [columns.index(a) for a in AXES]]
    if len(t_ms) < 3 or np.any(np.diff(t_ms) <= 0):
        raise ValueError(f'{path.name}: Zeitstempel muessen streng steigen; Aufnahme pruefen.')
    if not np.equal(raw, np.round(raw)).all() or np.any(raw < -2048) or np.any(raw > 2047):
        raise ValueError(f'{path.name}: keine vorzeichenbehafteten 12-bit-Rohwerte.')
    return Record(path.name, t_ms, raw, full_scale_g)

def timing(record):
    """Stichproben-SD (ddof=1); zwei verschiedene Frequenzmittel klar getrennt."""
    dt = np.diff(record.t_ms) / 1000.0
    fi = 1.0 / dt
    return {
        'N': len(record.t), 'Dauer_s': float(record.t[-1]),
        'dt_Mittel_ms': float(dt.mean() * 1000),
        'dt_SD_ms': float(dt.std(ddof=1) * 1000),
        'fs_eff_Hz': float((len(record.t)-1) / record.t[-1]),
        'fi_Mittel_Hz': float(fi.mean()), 'fi_SD_Hz': float(fi.std(ddof=1)),
        'dt_Min_ms': float(dt.min() * 1000), 'dt_Max_ms': float(dt.max() * 1000)
    }

def load_mobile(path):
    """Letzte monotone Aufnahme, erste zehn Sekunden nach ihrem ersten Zeitstempel."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 4 or not np.isfinite(data).all():
        raise ValueError('Mobile Datei: X, Y, Z in g und Zeit in ms erwartet.')
    starts = np.r_[0, np.flatnonzero(np.diff(data[:, 3]) <= 0) + 1]
    segment = data[starts[-1]:]
    if segment[-1, 3] - segment[0, 3] < 10000:
        raise ValueError('Letzte mobile Aufnahme kuerzer als zehn Sekunden.')
    window = segment[segment[:, 3] - segment[0, 3] <= 10000]
    return Record(Path(path).name, window[:, 3], window[:, :3], 8, 'g'), starts + 1

def butterworth_lowpass(values, cutoff_hz, fs, order=4):
    """Digitaler Butterworth-Tiefpass und Vorwärts-/Rückwärtsfilterung."""
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError('Grenzfrequenz muss zwischen null und der Nyquist-Frequenz liegen.')
    try:
        from scipy.signal import butter, filtfilt
    except ModuleNotFoundError as exc:
        if exc.name != 'scipy':
            raise
        return _butterworth_numpy(values, cutoff_hz, fs, order)
    b, a = butter(order, cutoff_hz / (fs / 2), btype='lowpass')
    return filtfilt(b, a, values, axis=0)


def _butterworth_numpy(values, cutoff_hz, fs, order):
    """Offline-Alternative: bilineare Transformation und ungerade Randfortsetzung."""
    k = np.arange(order)
    poles = np.exp(1j * np.pi * (2*k + order + 1)/(2*order))
    warped = np.tan(np.pi * cutoff_hz/fs)
    a = np.poly((1 + warped*poles)/(1 - warped*poles)).real
    b = np.poly(-np.ones(order))
    b *= a.sum()/b.sum()
    pad = 3*len(a)
    if len(values) <= pad:
        raise ValueError('Zu wenige Messwerte für die Filterung.')
    padded = np.concatenate((2*values[:1]-values[pad:0:-1], values,
                             2*values[-1:]-values[-2:-pad-2:-1]), axis=0)
    companion_t = np.zeros((order, order))
    companion_t[:, 0] = -a[1:]
    companion_t[:-1, 1:] = np.eye(order-1)
    zi = np.linalg.solve(np.eye(order)-companion_t, b[1:]-a[1:]*b[0])

    def run(samples):
        state = zi[:, None] * samples[0]
        output = np.empty_like(samples, dtype=float)
        for i, sample in enumerate(samples):
            output[i] = b[0]*sample + state[0]
            state = (np.vstack((state[1:], np.zeros_like(state[:1])))
                     + b[1:, None]*sample - a[1:, None]*output[i])
        return output

    return run(run(padded)[::-1])[::-1][pad:-pad]

def orientation_table(record):
    """Explizite ruhige Innenfenster; für die Mittelwerte der sechs Lagen."""
    windows = [('Z-',1,9), ('Z+',12,19), ('Y-',22,29),
               ('Y+',33,39), ('X-',42,49), ('X+',53,61)]
    rows = []
    for label, start, end in windows:
        mask = (record.t >= start) & (record.t <= end)
        if mask.sum() < 10:
            raise ValueError(f'Zu wenige Werte fuer {label}.')
        mean = record.g[mask].mean(axis=0)
        sd = record.g[mask].std(axis=0, ddof=1)
        j = AXES.index(label[0])
        target = 1 if label[1] == '+' else -1
        rows.append({'Lage':label, 'Start_s':start, 'Ende_s':end, 'N':int(mask.sum()),
                     **{f'{a}_Mittel_g':float(mean[k]) for k,a in enumerate(AXES)},
                     'Hauptachse_SD_g':float(sd[j]),
                     'Fehler_zum_Soll_g':float(mean[j]-target)})
    return rows

def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
