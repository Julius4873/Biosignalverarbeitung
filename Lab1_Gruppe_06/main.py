"""Auswertung und Diagramme für Labor 1."""


def main():
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plot
    from lab1 import (AXES, load_record, load_mobile, timing, butterworth_lowpass, orientation_table,
                      write_csv)

    BASE = Path(__file__).resolve().parent
    DATA = BASE / 'daten'
    RESULT = BASE / 'ergebnisse'
    RESULT.mkdir(exist_ok=True)

    # Raw counts; Empfindlichkeit = 2048 / Messbereich in g.
    a4 = load_record(DATA/'ACC_Data_Aufgabe1.4_2g.12.txt', ('X','Y','Z','t_ms'), 2)
    a5slow = load_record(DATA/'ACC_DATA_Aufgabe1.5_4g.12.txt', ('X','Y','Z','t_ms'), 4)
    a5fast = load_record(DATA/'ACC_Data_Aufgabe1.5_4g.100.txt', ('t_ms','X','Y','Z'), 4)
    a6 = load_record(DATA/'ACC_Data_Aufgabe1.6_2g.12.txt', ('t_ms','X','Y','Z'), 2)
    records = [a4, a5slow, a5fast, a6]
    inventory = [{'Datei':r.name, 'Messbereich_g':r.full_scale_g, **timing(r)} for r in records]
    write_csv(RESULT/'datenpruefung.csv', inventory)
    for row in inventory:
        print(f"{row['Datei']}: N={row['N']}, Dauer={row['Dauer_s']:.3f} s, fs_eff={row['fs_eff_Hz']:.4f} Hz")


    df = pd.DataFrame(a4.g, columns=['X', 'Y', 'Z'])
    df['Time'] = a4.t
    df.plot(x='Time', y=['X', 'Y', 'Z'], ylabel='Beschleunigung in g',
            xlabel='Zeit in s', figsize=[8, 6], title='Aufgabe 4: Beschleunigung bei ±2 g',
            xlim=(4, 8), linestyle='-')
    plot.savefig(RESULT/'aufgabe4.pdf', bbox_inches='tight')
    plot.savefig(RESULT/'aufgabe4.svg', bbox_inches='tight')
    plot.show()
    for j,a in enumerate(AXES):
        n = int(np.count_nonzero((a4.raw[:,j] == -2048) | (a4.raw[:,j] == 2047)))
        print(f'{a}: Min={a4.g[:,j].min():.4f} g, Max={a4.g[:,j].max():.4f} g, Grenzcodes={n}')

    fig, axes = plot.subplots(2, 1, figsize=[8, 6], sharex=True, sharey=True,
                             layout='constrained')
    for ax, record, title in zip(axes, (a4, a5fast), ('±2 g / 12,70 Hz', '±4 g / 101,49 Hz')):
        df = pd.DataFrame(record.g, columns=['X', 'Y', 'Z'])
        df['Time'] = record.t
        df.plot(x='Time', y=['X', 'Y', 'Z'], ax=ax, ylabel='Beschleunigung in g',
                xlabel='Zeit in s', title=title, xlim=(4, 8), ylim=(-4.5, 4.5), linestyle='-')
    fig.savefig(RESULT/'aufgabe5.pdf')
    fig.savefig(RESULT/'aufgabe5.svg')
    plot.show()
    for r in (a5slow,a5fast):
        print(r.name, 'max |a| =', round(float(np.abs(r.g).max()),4), 'g')

    df = pd.DataFrame(a6.g, columns=['X', 'Y', 'Z'])
    df['Time'] = a6.t
    df.plot(x='Time', y=['X', 'Y', 'Z'], ylabel='Beschleunigung in g',
            xlabel='Zeit in s', figsize=[8, 6], title='Aufgabe 6: Sechs Orientierungen',
            linestyle='-')
    plot.savefig(RESULT/'aufgabe6.pdf', bbox_inches='tight')
    plot.savefig(RESULT/'aufgabe6.svg', bbox_inches='tight')
    plot.show()
    orientations = orientation_table(a6)
    write_csv(RESULT/'orientierungen.csv', orientations)
    for row in orientations:
        means = ', '.join(f'{a}={row[a+"_Mittel_g"]:+.4f} g' for a in AXES)
        print(f"{row['Lage']}, {row['Start_s']}–{row['Ende_s']} s: {means}")

    cutoff_hz = 0.5
    filter_order = 4
    fs = timing(a6)["fs_eff_Hz"]
    filtered = butterworth_lowpass(a6.g, cutoff_hz, fs, filter_order)
    fig, axes = plot.subplots(3, 1, figsize=[8, 8], sharex=True, layout='constrained')
    for j, axis in enumerate(['X', 'Y', 'Z']):
        df = pd.DataFrame({'Time': a6.t, 'Ungefiltert': a6.g[:, j],
                           'Butterworth, 0,5 Hz': filtered[:, j]})
        df.plot(x='Time', y=['Ungefiltert', 'Butterworth, 0,5 Hz'], ax=axes[j],
                ylabel='Beschleunigung in g', xlabel='Zeit in s', title=axis,
                color=['tab:blue', 'tab:orange'], linestyle='-')
    fig.savefig(RESULT/'aufgabe7.pdf')
    fig.savefig(RESULT/'aufgabe7.svg')
    plot.show()
    print(f'Effektive Abtastrate: {fs:.4f} Hz')


    stats6 = timing(a6)
    for key,value in stats6.items(): print(f'{key}: {value:.6f}')
    write_csv(RESULT/'abtastfrequenz.csv', [{'Messung':'stationaer', **stats6}])
    dt_ms = np.diff(a6.t_ms)
    df = pd.DataFrame({'Time': a6.t[1:], 'Intervall in ms': dt_ms,
                       'Frequenz in Hz': 1000 / dt_ms})
    fig, axes = plot.subplots(2, 1, figsize=[8, 5], sharex=True, layout='constrained')
    df.plot(x='Time', y='Intervall in ms', ax=axes[0], ylabel='Intervall in ms',
            xlabel='Zeit in s', color='tab:blue', linestyle='-', legend=False)
    df.plot(x='Time', y='Frequenz in Hz', ax=axes[1], ylabel='Frequenz in Hz',
            xlabel='Zeit in s', color='tab:green', linestyle='-', legend=False)
    fig.savefig(RESULT/'aufgabe8.pdf')
    fig.savefig(RESULT/'aufgabe8.svg')
    plot.show()


    mobile, segment_starts = load_mobile(DATA/'ACC_Data_Aufgabe1.9_8g.200.txt')
    stats9 = timing(mobile)
    write_csv(RESULT/'vergleich_mobil.csv', [
        {'Messung':'stationaer', **stats6}, {'Messung':'mobil', **stats9}])
    print('Abschnittsstarts (Datenzeilen):', segment_starts)
    print('stationaer:', stats6)
    print('mobil:', stats9)
    print(f"Differenz fs_eff: {stats9['fs_eff_Hz']-stats6['fs_eff_Hz']:+.4f} Hz")
    dt_ms = np.diff(mobile.t_ms)
    df = pd.DataFrame({'Time': mobile.t[1:], 'Intervall in ms': dt_ms,
                       'Frequenz in Hz': 1000 / dt_ms})
    fig, axes = plot.subplots(2, 1, figsize=[8, 5], sharex=True, layout='constrained')
    df.plot(x='Time', y='Intervall in ms', ax=axes[0], ylabel='Intervall in ms',
            xlabel='Zeit in s', color='tab:blue', linestyle='-', legend=False)
    df.plot(x='Time', y='Frequenz in Hz', ax=axes[1], ylabel='Frequenz in Hz',
            xlabel='Zeit in s', color='tab:green', linestyle='-', legend=False)
    fig.savefig(RESULT/'aufgabe9.pdf')
    fig.savefig(RESULT/'aufgabe9.svg')
    plot.show()


    assert all(np.all(np.diff(r.t_ms)>0) for r in records)
    assert np.isclose(a6.ms2[0,2], a6.raw[0,2]/1024*9.80665)
    assert filtered.shape == a6.g.shape and np.isfinite(filtered).all()
    assert len(orientations) == 6
    assert len({r['Lage'] for r in orientations}) == 6
    assert 9.95 <= mobile.t[-1] <= 10 and np.all(np.diff(mobile.t_ms)>0)
    assert all((RESULT/f'aufgabe{k}.pdf').exists() for k in (4,5,6,7,8,9))
    summary = {'Gruppe':'06', 'Repository_Commit':'6d27ad9cef8c69c1a5f13fc96a5b74782a6ff0dc',
               'Daten':inventory, 'Orientierungen':orientations, 'Aufgabe8':stats6,
               'Aufgabe9':stats9}
    (RESULT/'auswertung.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Pruefungen 4–9 bestanden.')


if __name__ == "__main__":
    main()
