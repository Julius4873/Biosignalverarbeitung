"""Erstellt die Diagramme und Kennwerte für Labor 1."""
from pathlib import Path


def main():
    from pathlib import Path
    import json
    import numpy as np
    from lab1 import (AXES, COLORS, load_record, load_mobile, timing, butterworth_lowpass, orientation_table,
                      write_csv, save_plot, accel_panel, plot_timing)

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

    fig4 = save_plot(RESULT/'aufgabe4', 'Aufgabe 4: Beschleunigung, Bereich +/-2 g',
                     [accel_panel(a4, 'Rohdaten: 1024 Counts/g', 4, 8)])
    for j,a in enumerate(AXES):
        n = int(np.count_nonzero((a4.raw[:,j] == -2048) | (a4.raw[:,j] == 2047)))
        print(f'{a}: Min={a4.g[:,j].min():.4f} g, Max={a4.g[:,j].max():.4f} g, Grenzcodes={n}')

    panels = [accel_panel(a4, '+/-2 g; langsame Aufnahme', 4, 8),
              accel_panel(a5fast, '+/-4 g; schnellere Aufnahme', 4, 8)]
    # Gleiche y-Skala ermöglicht direkten Amplitudenvergleich.
    for p in panels: p['ylim'] = (-4.5, 4.5)
    fig5 = save_plot(RESULT/'aufgabe5', 'Aufgabe 5: Messbereich und Aufzeichnungsrate', panels)
    for r in (a5slow,a5fast):
        print(r.name, 'max |a| =', round(float(np.abs(r.g).max()),4), 'g')

    fig6 = save_plot(RESULT/'aufgabe6', 'Aufgabe 6: gesamte zusammenhaengende Aufnahme',
                     [accel_panel(a6, 'Reihenfolge: -Z, +Z, -Y, +Y, -X, +X')], panel_height=210)
    orientations = orientation_table(a6)
    write_csv(RESULT/'orientierungen.csv', orientations)
    for row in orientations:
        means = ', '.join(f'{a}={row[a+"_Mittel_g"]:+.4f} g' for a in AXES)
        print(f"{row['Lage']}, {row['Start_s']}–{row['Ende_s']} s: {means}")

    cutoff_hz = 0.5
    filter_order = 4
    fs = timing(a6)["fs_eff_Hz"]
    filtered = butterworth_lowpass(a6.g, cutoff_hz, fs, filter_order)
    panels = []
    for j,a in enumerate(AXES):
        panels.append({'title':a+'-Achse', 'xlim':(0,a6.t[-1]), 'ylim':(-1.35,1.35), 'ylabel':'a (g)',
                       'curves':[(a6.t,a6.g[:,j],'Ungefiltert','#586580',None),
                                 (a6.t,filtered[:,j],'Butterworth, 0.5 Hz',COLORS[j],None)]})
    fig7 = save_plot(RESULT/'aufgabe7', 'Aufgabe 7: Rohdaten und Butterworth-Tiefpass', panels)
    print(f"Effektive Abtastrate: {fs:.4f} Hz")

    stats6 = timing(a6)
    for key,value in stats6.items(): print(f'{key}: {value:.6f}')
    write_csv(RESULT/'abtastfrequenz.csv', [{'Messung':'stationaer', **stats6}])
    fig8 = plot_timing(a6, RESULT/'aufgabe8', 'Aufgabe 8: Abtastintervalle und lokale Frequenz')

    mobile, segment_starts = load_mobile(DATA/'ACC_Data_Aufgabe1.9_8g.200.txt')
    stats9 = timing(mobile)
    write_csv(RESULT/'vergleich_mobil.csv', [
        {'Messung':'stationaer', **stats6}, {'Messung':'mobil', **stats9}])
    print('Abschnittsstarts (Datenzeilen):', segment_starts)
    print('stationaer:', stats6)
    print('mobil:', stats9)
    print(f"Differenz fs_eff: {stats9['fs_eff_Hz']-stats6['fs_eff_Hz']:+.4f} Hz")
    fig9 = plot_timing(mobile, RESULT/'aufgabe9', 'Aufgabe 9: mobile Abtastfrequenz')

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
