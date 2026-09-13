import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


error_123 = math.sqrt(0.028**2 + 0.0062**2)   # 0.62 %
error_456 = math.sqrt(0.028**2 + 0.0096**2)  # 0.96 %

kQ=0.665# alter Wert 0.6748

zVerschiebung = -0.526 #in mm   
Pixel = 8.5
zDetektorebene = zVerschiebung + 0.1*Pixel #Korrektur in mm
# print(zDetektorebene)


#Dreisatz: zPhantom/zDetektor = StrahlPhantom(SID)/StrahlDetektor(SDD)
Sid = 441
Sdd = 507
zPhantomebene = zDetektorebene * (Sid/Sdd)
# print(zPhantomebene, " in mm")
zPhantomebene_cm = zPhantomebene / 10

# radiuspixel = 13.5
radiuspixel = 127.5-110.985 
kollimiertesStrahlzentrum = Pixel * 0.1 * (Sid / Sdd) + 0.1 * radiuspixel
# print(kollimiertesStrahlzentrum, " in mm")






# PTW FArmer Chamber 30013
# ============================
# Kalibrierfaktoren
# ============================

Nk = 5.539e7      # Gy/C
kQ2 = 0.901 #alter Wert 0.8921     #0.8917 früher          # <-- hier später den passenden Wert einsetzen
kQ2_error = 0.011

kTp1 = 1.056913754
kTp23 = 1.055732087

kTp_error = 0.007

yrho = 1.0224
yrho_error = 0.0036

p=1.002
p_error=0.015


ptw_error =math.sqrt((kQ2_error/kQ2)**2+(yrho_error/yrho)**2+(p_error/p)**2+kTp_error**2)
# ============================
# Messwerte [pC]
# ============================

# kleines Phantom - Center
small_center = [984.4*kTp1, 983.4*kTp23, 972.6*kTp23] 

# kleines Phantom - Off
small_off = [920.1*kTp1, 923.0*kTp23, 915.0*kTp23]

# großes Phantom - Center
large_center = [940.4*kTp1, 940.8*kTp23, 945.6*kTp23]

# großes Phantom - Off
large_off = [876.6*kTp1, 873.2*kTp23, 877.2*kTp23]

# ============================
# Umrechnung pC -> mGy
# ============================

def dose(pc):
    return Nk * pc * 1e-12 * kQ2 * 1000 * yrho * p  # mGy

# ============================
# Ergebnisse
# ============================
# ============================
# PTW-Dosen und Mittelwerte
# ============================

# Einzelwerte von pC in mGy umrechnen
ptw_small_center_doses = np.array([dose(x) for x in small_center])
ptw_small_off_doses    = np.array([dose(x) for x in small_off])
ptw_large_center_doses = np.array([dose(x) for x in large_center])
ptw_large_off_doses    = np.array([dose(x) for x in large_off])

# Mittelwerte der jeweils drei Messungen
ptw_small_center_avg = np.mean(ptw_small_center_doses)
ptw_small_off_avg    = np.mean(ptw_small_off_doses)
ptw_large_center_avg = np.mean(ptw_large_center_doses)
ptw_large_off_avg    = np.mean(ptw_large_off_doses)

# Unsicherheit des PTW-kQ: ±kQ_error % des Mittelwertes
ptw_small_center_err = ptw_error * ptw_small_center_avg
ptw_small_off_err    = ptw_error * ptw_small_off_avg
ptw_large_center_err = ptw_error * ptw_large_center_avg
ptw_large_off_err    = ptw_error * ptw_large_off_avg

# ============================
# Ergebnisse ausgeben
# ============================

print("\nKleines Phantom – Center")
# print("Einzeldosen:", ptw_small_center_doses)
print(f"Mittelwert: {ptw_small_center_avg:.3f} ± "
      f"{ptw_small_center_err:.3f} mGy")

print("\nKleines Phantom – Off")
# print("Einzeldosen:", ptw_small_off_doses)
print(f"Mittelwert: {ptw_small_off_avg:.3f} ± "
      f"{ptw_small_off_err:.3f} mGy")

print("\nGroßes Phantom – Center")
# print("Einzeldosen:", ptw_large_center_doses)
print(f"Mittelwert: {ptw_large_center_avg:.3f} ± "
      f"{ptw_large_center_err:.3f} mGy")

print("\nGroßes Phantom – Off")
# print("Einzeldosen:", ptw_large_off_doses)
print(f"Mittelwert: {ptw_large_off_avg:.3f} ± "
      f"{ptw_large_off_err:.3f} mGy")






#dünn Zentral
pos_1 = np.array([-2.7,-2.1,-1.5,-0.9,-0.3,0.3,0.9,1.5,2.1,2.7])*10  # cm
dose_1 = np.array([2.2,4.06,33.39,78.97,82.28,80.48,79.82,76.27,52.51,4.69]) * kQ

pos_5 = np.array([-3,-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4,3.0]) *10 
dose_5 = np.array([1.88,2.9,6.13,70.07,80.65,80.54,80.38,78.08,73.01,16.05,2.92]) * kQ

#dünn Off
pos_2 = np.array([-2.7,-2.1,-1.5,-0.9,-0.3,0.3,0.9,1.5,2.1,2.7]) *10 
dose_2 = np.array([2.04,3.22,7.18,66.39,78.17,78.88,76.83,71.58,13.72,3]) * kQ

pos_6 = np.array([-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4]) *10 
dose_6 = np.array([2.32,4.38,30.48,78.12,80.05,78.31,74.67,46.75,5]) * kQ

#dick Zentral
pos_4 = np.array([-3,-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4,3.0]) *10 
dose_4 = np.array([2.18,3.47,7.39,67.17,78.15,79.49,79.44,76.3,70.13,16.55,3.76]) * kQ

#dick Off
pos_3 = np.array([-3,-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4,3.0]) *10 
dose_3 = np.array([1.83,2.8,5.19,29.55,73.23,76.56,75.79,73.05,46.52,5.57,2.69]) * kQ


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

#Normal
def plot_profiles(datasets, title, suptitle):
    plt.figure(figsize=(10, 6))
    
    for data in datasets:
        pos, dose, label, error = data
        
        plt.errorbar(
            pos,
            dose,
            yerr=dose*error,
            label=label,
            fmt='o',
            capsize=5,
            capthick=1.5,
            elinewidth=1.5
        )

    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis pro Tomographie [mGy]")
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.show()

#gedreht
def plot_profiles_rotated(datasets, title, suptitle):
    plt.figure(figsize=(10, 6))
    
    for data in datasets:
        pos, dose, label, error = data
        
        plt.errorbar(
            dose,                # x = Dosis
            pos,                 # y = Position
            xerr=dose*error,     # Fehler in x-Richtung
            label=label,
            fmt='o-',
            capsize=5,
            capthick=1.5,
            elinewidth=1.5
        )

    plt.xlabel("Dosis pro Tomographie [mGy]", loc='right')
    plt.ylabel("vertikale Position [mm]")
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.show()

#mit plot
def plot_profiles_with_fit(datasets, title, suptitle, z_phantom_cm):
    plt.figure(figsize=(10, 6))
    
    fit_colors = ['blue', 'orange']
    
    for i, data in enumerate(datasets):
        pos, dose, label, error = data
        
        # Messwerte
        plt.errorbar(
            pos,
            dose,
            yerr=dose*error,
            label=label,
            fmt='o',
            capsize=5
        )

        try:
            # Nur relevante Messpunkte für den Gauß-Fit
            mask = dose > (np.max(dose) * 0.1)
            fit_pos = pos[mask]
            fit_dose = dose[mask]

            # Startwerte
            p0 = [
                np.max(dose),
                fit_pos[np.argmax(fit_dose)],
                10.0
            ]

            weights = 1.0 / (fit_dose + 1e-6)

            # Gauß-Fit
            popt, _ = curve_fit(
                gaus,
                fit_pos,
                fit_dose,
                p0=p0,
                sigma=weights
            )

            a, x0, sigma = popt
            
            current_color = fit_colors[i % len(fit_colors)]

            x_range = np.linspace(
                min(pos) - 5.0,
                max(pos) + 5.0,
                500
            )

            # Gauß-Funktion
            plt.plot(
                x_range,
                gaus(x_range, *popt),
                '--',
                color=current_color,
                alpha=0.9,
                label=f'Gauss-Fit ({label})'
            )
            
            # Gauß-Mittelpunkt
            plt.axvline(
                x=x0,
                color=current_color,
                linestyle='-',
                linewidth=1
            )
            
            # Beschriftung unterhalb der x-Achse
            plt.annotate(f'Gauss-Fitmittelpunkt: {x0:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (i * 18)), textcoords='offset points', va='top', ha='right', color=current_color, fontweight='bold')
            
        except Exception as e:
            print(f"Fit für {label} fehlgeschlagen: {e}")

    # Strahlmittelpunkt
    plt.axvline(
        x=z_phantom_cm,
        color='darkgreen',
        linestyle='-',
        linewidth=2.5,
        label='Strahlmittelpunkt $S_0$'
    )

    plt.annotate(f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (len(datasets) * 18)), textcoords='offset points', va='top', ha='right', color='darkgreen', fontweight='bold')

    # Kollimatormittelpunkt
    plt.axvline(
        x=kollimiertesStrahlzentrum,
        color='m',
        linestyle='-',
        linewidth=2,
        label='Kollimatormittelpunkt $K_0$'
    )

    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - ((len(datasets) + 1) * 18)), textcoords='offset points', va='top', ha='right', color='m', fontweight='bold')

    # Achsenbeschriftungen
    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis pro Tomographie [mGy]")

    # Titel und Untertitel
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)

    # Legende und Gitter
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    # Platz für Beschriftungen unterhalb der x-Achse
    plt.subplots_adjust(bottom=0.22)

    plt.show()



# mit Konvolution
def plot_profiles_konvolution(datasets, title, suptitle, z_phantom_cm):
    from scipy.signal import fftconvolve

    plt.figure(figsize=(10, 6))
    
    fit_colors = ['blue', 'orange']

    # Rechteck (zentriert bei 0)
    def rect(x, width):
        return np.where(np.abs(x) <= width/2, 1.0, 0.0)

    # Gauß (zentriert bei 0)
    def gauss(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2))

    # Modell: Rechteck ⊗ Gauß, anschließend um x0 verschoben
    def rect_gauss_conv(x, a, x0, sigma, width):
        dx = x[1] - x[0]
        
        x_kernel = np.linspace(-50, 50, 2000)
        
        r = rect(x_kernel, width)
        g = gauss(x_kernel, sigma)

        conv = fftconvolve(r, g, mode='same') * dx
        conv = conv / np.max(conv)

        return a * np.interp(x - x0, x_kernel, conv)

    for i, data in enumerate(datasets):
        pos, dose, label, error = data
        
        # Messwerte
        plt.errorbar(
            pos,
            dose,
            yerr=dose*error,
            label=label,
            fmt='o',
            capsize=5
        )

        try:
            # Startwerte
            p0 = [
                np.max(dose),          # a
                pos[np.argmax(dose)],  # x0
                3.0,                   # sigma
                30.0                    # width
            ]

            bounds = (
                [0, -50, 0.5, 10.0],
                [200, 50, 20.0, 100.0]
            )

            # Konvolutions-Fit
            popt, _ = curve_fit(
                rect_gauss_conv,
                pos,
                dose,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )

            a, x0, sigma, width = popt

            current_color = fit_colors[i % len(fit_colors)]

            x_range = np.linspace(
                min(pos) - 5.0,
                max(pos) + 5.0,
                500
            )
            
            y_fit = rect_gauss_conv(x_range, *popt)

            # Konvolutions-Fit
            plt.plot(
                x_range,
                y_fit,
                '--',
                color=current_color,
                alpha=0.9,
                label=f'Multiplikations-Fit ({label})'
            )

            # Fitmittelpunkt
            x_max = x0

            plt.axvline(
                x=x_max,
                color=current_color,
                linestyle='-',
                linewidth=1
            )

            # Beschriftung unterhalb der x-Achse
            plt.annotate(f'Multiplikations-Fitmittelpunkt: {x_max:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (i * 18)), textcoords='offset points', va='top', ha='right', color=current_color, fontweight='bold')

        except Exception as e:
            print(f"Fit für {label} fehlgeschlagen: {e}")

    # Strahlmittelpunkt
    plt.axvline(
        x=z_phantom_cm,
        color='darkgreen',
        linestyle='-',
        linewidth=2.5,
        label='Strahlmittelpunkt $S_0$'
    )

    plt.annotate(f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (len(datasets) * 18)), textcoords='offset points', va='top', ha='right', color='darkgreen', fontweight='bold')

    # Kollimatormittelpunkt
    plt.axvline(
        x=kollimiertesStrahlzentrum,
        color='m',
        linestyle='-',
        linewidth=2,
        label='Kollimatormittelpunkt $K_0$'
    )

    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - ((len(datasets) + 1) * 18)), textcoords='offset points', va='top', ha='right', color='m', fontweight='bold')

    # Achsenbeschriftungen
    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis pro Tomographie [mGy]")

    # Titel und Untertitel
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)

    # Legende und Gitter
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    # Platz für Beschriftungen unterhalb der x-Achse
    plt.subplots_adjust(bottom=0.22)

    plt.show()

# mit erf (Abgerundete Stufenfkt)(Gaus mal Rechteck)
def plot_profiles_erf(datasets,title,suptitle,z_phantom_cm,ptw_avg=None,ptw_x0=0,
                      ptw_length=23.0,      # mm = 23 mm
                      ptw_rel_err=0.011,   # 2.5 %
                      ptw_label="PTW"):
    

    plt.figure(figsize=(10, 6))
    
    fit_colors = ['blue', 'orange']

    # Rechteck ⊗ Gauß analytisch
    def rect_gauss_erf(x, A, x0, sigma, width):
        return (A/2) * (
            erf((x - x0 + width/2) / (np.sqrt(2)*sigma)) -
            erf((x - x0 - width/2) / (np.sqrt(2)*sigma))
        )
    # def find_xmax(func, popt, x_min, x_max):
    #     f = lambda x: -func(x, *popt)  # minus für Maximum
    #     res = minimize_scalar(f, bounds=(x_min, x_max), method='bounded')
    #     return res.x

    for i, data in enumerate(datasets):
        pos, dose, label, error = data
        
        plt.errorbar(pos, dose, yerr=dose*error, label=label, fmt='o', capsize=5)

        try:
            p0 = [
                np.max(dose),          # A
                pos[np.argmax(dose)],  # x0
                5.0,                   # sigma (Kantenbreite)
                30.0                    # width (Feldbreite)
            ]

            bounds = (
                [30, -10.0, 0.5, 10.0],
                [100, 16.0, 20.0, 100.0]
            )

            popt, _ = curve_fit(
                rect_gauss_erf,
                pos,
                dose,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )


            A, x0, sigma, width = popt

            # -------------------------------------------------
            # Plateaubreite (FWHM) und Halbwertspunkte
            # -------------------------------------------------
            y_half   = rect_gauss_erf(x0, *popt) / 2
            x_half_l = x0 - width / 2
            x_half_r = x0 + width / 2

            print(f"{label}: FWHM = {width:.3f} mm, "
                  f"Halbwertspunkte bei {x_half_l:.3f} / {x_half_r:.3f} mm, "
                  f"Halbwertsdosis = {y_half:.3f} mGy")

            

            current_color = fit_colors[i % len(fit_colors)]
            x_range = np.linspace(min(pos)-5.0, max(pos)+5.0, 500)
            
            y_fit = rect_gauss_erf(x_range, *popt)

            plt.plot(
                x_range,
                y_fit,
                '--',
                color=current_color,
                alpha=0.9,
                label=f'ERF-Fit ({label})'
            )

            # Maximumn
            #x_range[np.argmax(y_fit)]
            x_max = x0 #find_xmax(rect_gauss_erf, popt, min(pos), max(pos))

            plt.axvline(x=x_max, color=current_color, linestyle='-', linewidth=1)
            plt.annotate(f'erf-Fitmittelpunkt: {x_max:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (i * 18)), textcoords='offset points', va='top', ha='right', color=current_color, fontweight='bold')

            plateau = np.max(y_fit)

            print(f"Maximale Fit-Dosis = {plateau:.2f} mGy")

        except Exception as e:
            print(f"Fit für {label} fehlgeschlagen: {e}")

    # Phantomposition
    plt.axvline(x=z_phantom_cm, color='darkgreen', linestyle='-', linewidth=2.5, label='Strahlmittelpunkt $S_0$')
    plt.annotate(f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (len(datasets) * 18)), textcoords='offset points', va='top', ha='right', color='darkgreen', fontweight='bold')
    plt.axvline(kollimiertesStrahlzentrum, color='m', linewidth=2, label='Kollimatormittelpunkt $K_0$')
    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - ((len(datasets) + 1) * 18)), textcoords='offset points', va='top', ha='right', color='m', fontweight='bold')

    # # Phantom position auf englisch
    # plt.axvline(x=z_phantom_cm, color='darkgreen', linestyle='-', linewidth=2.5)
    # plt.annotate(f'Beam center: {z_phantom_cm:.3f} mm', xy=(z_phantom_cm, 0), xycoords=('data', 'axes fraction'), xytext=(0, -56), textcoords='offset points', va='top', ha='center', color='darkgreen', fontweight='bold')
    # plt.axvline(kollimiertesStrahlzentrum, color='m', linewidth=2,)
    # plt.annotate(f'Collimated beam center: {kollimiertesStrahlzentrum:.3f} mm', xy=(kollimiertesStrahlzentrum, 0), xycoords=('data', 'axes fraction'), xytext=(0, -70), textcoords='offset points', va='top', ha='center', color='m', fontweight='bold')
# -------------------------------------------------
# PTW-Linie mit Unsicherheit
# -------------------------------------------------
    if ptw_avg is not None and ptw_x0 is not None:
    
        ptw_yerr = ptw_avg * ptw_rel_err
        x_left = ptw_x0 - ptw_length / 2
        x_right = ptw_x0 + ptw_length / 2

        # horizontale Linie über 23 mm
        plt.hlines(
            y=ptw_avg,
            xmin=x_left,
            xmax=x_right,
            colors='red',
            linewidth=2.5,
            label=f'{ptw_label}'
        )

        # vertikaler Fehlerbalken in der Mitte
        plt.errorbar(
            ptw_x0,
            ptw_avg,
            yerr=ptw_yerr,
            fmt='none',
            ecolor='red',
            elinewidth=2,
            capsize=6
        )

        # optional: schattiertes Fehlerband über die ganze Länge
        plt.fill_between(
            [x_left, x_right],
            [ptw_avg - ptw_yerr, ptw_avg - ptw_yerr],
            [ptw_avg + ptw_yerr, ptw_avg + ptw_yerr],
            color='red',
            alpha=0.15
        )

        print(f"{ptw_label}:")
        print(f"  Mittelwert = {ptw_avg:.2f} mGy")
        print(f"  Fehler     = ±{ptw_yerr:.2f} mGy")
        print(f"  x-Bereich  = [{x_left:.3f}, {x_right:.3f}] mm")


    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis pro Tomographie [mGy]")
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.subplots_adjust(bottom=0.22)
    plt.show()

def plot_profiles_erf_mitFehler(datasets, title, suptitle, z_phantom_cm):

    plt.figure(figsize=(10, 6))
    
    fit_colors = ['blue', 'orange', '#1E90FF', '#FFB347']


    # ---------------------------------------------
    # Rechteck ⊗ Gauß analytisch
    # ---------------------------------------------
    def rect_gauss_erf(x, A, x0, sigma, width):
        return (A/2) * (
            erf((x - x0 + width/2) / (np.sqrt(2)*sigma)) -
            erf((x - x0 - width/2) / (np.sqrt(2)*sigma))
        )


    fit_colors = ['blue', 'orange', '#1E90FF', '#FFB347']

    x_error    = 3.0          # halbe Breite -> 6 mm Gesamtlänge
    grain_h    = 4.0          # Höhe der Füllung in pt
    grain_face = '#7CB342'    # Grasgrün
    grain_edge = '#14646E'    # Petrol
    grain_ring = 1.2          # Randbreite in pt (rundum gleich)

    grains = []               # sammelt (pos, dose) je Datensatz

    for i, data in enumerate(datasets):

        pos, dose, label, error = data

        # -------------------------------------------------
        # Fehlerbalken
        # -------------------------------------------------
        y_error = dose * error

        grains.append((np.asarray(pos, dtype=float),
                       np.asarray(dose, dtype=float)))

        plt.errorbar(
            pos,
            dose,
            yerr=y_error,
            label=label,
            fmt='o',
            markersize=grain_h-0.5,
            capsize=5,
            zorder=3
        )

        try:

            # -------------------------------------------------
            # Normaler ERF-Fit
            # -------------------------------------------------
            p0 = [
                np.max(dose),
                pos[np.argmax(dose)],
                5.0,
                30.0
            ]

            bounds = (
                [30, -10.0, 0.5, 10.0],
                [100, 16.0, 20.0, 100.0]
            )

            popt, _ = curve_fit(
                rect_gauss_erf,
                pos,
                dose,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )

            A, x0, sigma, width = popt

            current_color = fit_colors[i % 2]
            error_color = fit_colors[(i % 2) + 2]

            x_range = np.linspace(
                min(pos) - 5.0,
                max(pos) + 5.0,
                500
            )

            y_fit = rect_gauss_erf(x_range, *popt)

            # Hauptfit
            plt.plot(
                x_range,
                y_fit,
                '--',
                color=current_color,
                alpha=0.9,
                linewidth=2,
                label=f'ERF-Fit ({label})'
            )

            # Fitmittelpunkt
            plt.axvline(
                x=x0,
                color=current_color,
                linestyle='-',
                linewidth=1.5
            )

            # Beschriftung unterhalb der x-Achse
            plt.annotate(f'erf-Fitmittelpunkt: {x0:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (i * 18)), textcoords='offset points', va='top', ha='right', color=current_color, fontweight='bold')


            # -------------------------------------------------
            # Min-/Max-Fits mit x-Fehler
            # -------------------------------------------------

            # Punkte maximal links
            pos_min = pos - x_error

            # Punkte maximal rechts
            pos_max = pos + x_error

            # Min-Fit
            popt_min, _ = curve_fit(
                rect_gauss_erf,
                pos_min,
                dose,
                p0=popt,
                bounds=bounds,
                maxfev=20000
            )

            # Max-Fit
            popt_max, _ = curve_fit(
                rect_gauss_erf,
                pos_max,
                dose,
                p0=popt,
                bounds=bounds,
                maxfev=20000
            )

            y_fit_min = rect_gauss_erf(x_range, *popt_min)
            y_fit_max = rect_gauss_erf(x_range, *popt_max)

            # Grenzfits
            plt.plot(
                x_range,
                y_fit_min,
                ':',
                color=error_color,
                alpha=0.9,
                linewidth=2
            )

            plt.plot(
                x_range,
                y_fit_max,
                ':',
                color=error_color,
                alpha=0.9,
                linewidth=2
            )

            # Mittelpunktlinien der Grenzfits
            x0_min = popt_min[1]
            x0_max = popt_max[1]

            plt.axvline(
                x=x0_min,
                color=error_color,
                linestyle=':',
                linewidth=1.5
            )

            plt.axvline(
                x=x0_max,
                color=error_color,
                linestyle=':',
                linewidth=1.5
            )

        except Exception as e:
            print(f"Fit für {label} fehlgeschlagen: {e}")


    # -------------------------------------------------
    # Strahlmittelpunkt
    # -------------------------------------------------
    plt.axvline(
        x=z_phantom_cm,
        color='darkgreen',
        linestyle='-',
        linewidth=2.5,
        label='Strahlmittelpunkt $S_0$'
    )

    plt.annotate(f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (len(datasets) * 18)), textcoords='offset points', va='top', ha='right', color='darkgreen', fontweight='bold')


    # -------------------------------------------------
    # Kollimatormittelpunkt
    # -------------------------------------------------
    plt.axvline(
        x=kollimiertesStrahlzentrum,
        color='m',
        linestyle='-',
        linewidth=2,
        label='Kollimatormittelpunkt $K_0$'
    )

    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - ((len(datasets) + 1) * 18)), textcoords='offset points', va='top', ha='right', color='m', fontweight='bold')


    # -------------------------------------------------
    # Achsenbeschriftungen
    # -------------------------------------------------
    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis pro Tomographie [mGy]")


    # -------------------------------------------------
    # Titel und Untertitel
    # -------------------------------------------------
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)


    # -------------------------------------------------
    # Legende und Gitter
    # -------------------------------------------------
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)


    # Platz für Beschriftungen unterhalb der x-Achse
    plt.subplots_adjust(bottom=0.22)

    # -------------------------------------------------
    # x-Fehler als Zylinder: Rechteck, außen exakt 6 mm
    # -------------------------------------------------
    ax = plt.gca()
    ax.set_xlim(ax.get_xlim())          # Grenzen einfrieren
    ax.set_ylim(ax.get_ylim())
    plt.gcf().canvas.draw()

    # 1 pt in Datenkoordinaten (x-Richtung)
    inv = ax.transData.inverted()
    x_a, _ = inv.transform((0, 0))
    x_b, _ = inv.transform((plt.gcf().dpi / 72.0, 0))
    d = grain_ring * (x_b - x_a)        # seitliche Randbreite in mm

    for p_arr, d_arr in grains:
        for xi, yi in zip(p_arr, d_arr):
            # Außenrechteck: exakt 6 mm breit
            plt.plot(
                [xi - x_error, xi + x_error],
                [yi, yi],
                color=grain_edge,
                linewidth=grain_h + 2*grain_ring,
                solid_capstyle='butt',
                zorder=2.4
            )
            # Füllung: rundum um grain_ring kleiner
            plt.plot(
                [xi - x_error + d, xi + x_error - d],
                [yi, yi],
                color=grain_face,
                linewidth=grain_h,
                solid_capstyle='butt',
                zorder=2.5
            )

    plt.show()



def plot_profiles_exakt(datasets, title, suptitle, z_phantom_cm):

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange']

    # -------------------------------------------------------
    # FLEXIBLES STRAHLMODELL (alles steuerbar)
    # -------------------------------------------------------
    def beam_model(x,
                   A, x0,
                   width,
                   sigma_center,
                   sigma_left, sigma_right,
                   edge_left, edge_right,
                   tail_left, tail_right,
                   floor_left, floor_right):

        dx = x - x0

        # -------- Plateau (weiches Rechteck) --------
        plateau = 0.5 * (
            erf((dx + width/2) / (np.sqrt(2)*sigma_left)) -
            erf((dx - width/2) / (np.sqrt(2)*sigma_right))
        )

        # -------- zentrale Rundung --------
        center = np.exp(-(dx**2) / (2 * sigma_center**2))

        # -------- asymmetrischer Randabfall --------
        edge = np.where(
            dx < -width/2,
            np.exp(edge_left * (dx + width/2)),
            np.where(
                dx > width/2,
                np.exp(-edge_right * (dx - width/2)),
                1.0
            )
        )

        # -------- asymmetrischer Scatter-Untergrund --------
        floor = np.where(
            dx < 0,
            floor_left * np.exp(dx / tail_left),
            floor_right * np.exp(-dx / tail_right)
        )

        return A * (plateau * edge + 0.25 * center) + floor

    # -------------------------------------------------------
    # PLOTTING + FIT
    # -------------------------------------------------------
    for i, data in enumerate(datasets):

        pos, dose, label, error = data

        plt.errorbar(pos, dose, yerr=dose*error, fmt='o', capsize=4, label=label)

        try:
            # ---------------- initial guess ----------------
            p0 = [
                np.max(dose),                 # A
                pos[np.argmax(dose)],         # x0
                3.0,                          # width
                0.4,                          # sigma_center

                0.3, 0.3,                    # sigma L/R
                0.8, 0.8,                    # edge L/R

                2.0, 2.0,                    # tail L/R
                1.0, 1.0                     # floor L/R
            ]

            bounds = (                                                   #1/Breite des Falls    Abfalll außen   
                #A,     x0, Breit-Plateau   sigma-Mitte s-Links s-Rechts    Links   Rechts      Links   Rechts  Höhe Links  Höhe Rechts
                [0,     -10,0.5,            0.05,       0.01,   0.01,       0.01,   0.01,       0.1,    0.1,    0,          0],
                [200,   10, 10,             3.0,        5.0,    5.0,        10.0,   10.0,       20.0,   20.0,   50.0,       50.0]
            )

            popt, _ = curve_fit(
                beam_model,
                pos,
                dose,
                p0=p0,
                bounds=bounds,
                maxfev=50000
            )

            x_fit = np.linspace(min(pos)-0.5, max(pos)+0.5, 800)
            y_fit = beam_model(x_fit, *popt)

            plt.plot(x_fit, y_fit, '--', color=colors[i % len(colors)],
                     label=f'Fit {label}')

            # echtes Maximum
            x_max = x_fit[np.argmax(y_fit)]
            plt.axvline(x_max, color=colors[i % len(colors)], linewidth=1)
            plt.annotate(f'{x_max:.3f}', xy=(x_max, 0), xycoords=('data', 'axes fraction'), xytext=(0, -20 - (i * 18)), textcoords='offset points', va='top', ha='center', color=colors[i % len(colors)], fontweight='bold')

        except Exception as e:
            print(f"Fit fehlgeschlagen ({label}):", e)

    # -------------------------------------------------------
    # Phantomlinie
    # -------------------------------------------------------
    plt.axvline(z_phantom_cm, color='darkgreen', linewidth=2, label='Strahlmittelpunkt $S_0$')
    plt.annotate(f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f}', xy=(z_phantom_cm, 0), xycoords=('data', 'axes fraction'), xytext=(0, -56), textcoords='offset points', va='top', ha='center', color='darkgreen', fontweight='bold')
    plt.axvline(kollimiertesStrahlzentrum, color='m', linewidth=2, label='Kollimator')
    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f}', xy=(kollimiertesStrahlzentrum, 0), xycoords=('data', 'axes fraction'), xytext=(0, -70), textcoords='offset points', va='top', ha='center', color='m', fontweight='bold')
    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis [mGy]")
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)
    plt.grid()
    plt.legend()
    plt.show()

def plot_profiles_exakt_mitFehler(datasets, title, suptitle, z_phantom_cm):

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', '#1E90FF', '#FFB347']

    # -------------------------------------------------------
    # FLEXIBLES STRAHLMODELL
    # -------------------------------------------------------
    def beam_model(x,
                   A, x0,
                   width,
                   sigma_center,
                   sigma_left, sigma_right,
                   edge_left, edge_right,
                   tail_left, tail_right,
                   floor_left, floor_right):

        dx = x - x0

        # -------- Plateau --------
        plateau = 0.5 * (
            erf((dx + width/2) / (np.sqrt(2)*sigma_left)) -
            erf((dx - width/2) / (np.sqrt(2)*sigma_right))
        )

        # -------- zentrale Rundung --------
        center = np.exp(-(dx**2) / (2 * sigma_center**2))

        # -------- asymmetrischer Rand --------
        edge = np.where(
            dx < -width/2,
            np.exp(edge_left * (dx + width/2)),
            np.where(
                dx > width/2,
                np.exp(-edge_right * (dx - width/2)),
                1.0
            )
        )

        # -------- asymmetrischer Untergrund --------
        floor = np.where(
            dx < 0,
            floor_left * np.exp(dx / tail_left),
            floor_right * np.exp(-dx / tail_right)
        )

        return A * (plateau * edge + 0.25 * center) + floor

    # -------------------------------------------------------
    # PLOTTING + FIT
    # -------------------------------------------------------
    for i, data in enumerate(datasets):

        pos, dose, label, error = data

        # ---------------------------------------------------
        # FEHLERBALKEN
        # ---------------------------------------------------
        x_error = 3.0
        y_error = dose * error

        plt.errorbar(
            pos,
            dose,
            xerr=x_error,
            yerr=y_error,
            fmt='o',
            capsize=4,
            label=label
        )

        try:

            # ---------------------------------------------------
            # STARTWERTE
            # ---------------------------------------------------
            p0 = [
                np.max(dose),                 # A
                pos[np.argmax(dose)],         # x0
                3.0,                          # width
                0.4,                          # sigma_center

                0.3, 0.3,                    # sigma L/R
                0.8, 0.8,                    # edge L/R

                2.0, 2.0,                    # tail L/R
                1.0, 1.0                     # floor L/R
            ]

            # ---------------------------------------------------
            # BOUNDS
            # ---------------------------------------------------
            bounds = (
                [0,     -10, 0.5, 0.05,
                 0.01, 0.01,
                 0.01, 0.01,
                 0.1, 0.1,
                 0, 0],

                [200,   10, 10, 3.0,
                 5.0, 5.0,
                 10.0, 10.0,
                 20.0, 20.0,
                 50.0, 50.0]
            )

            # ---------------------------------------------------
            # NORMALER FIT
            # ---------------------------------------------------
            popt, _ = curve_fit(
                beam_model,
                pos,
                dose,
                p0=p0,
                bounds=bounds,
                maxfev=50000
            )

            x_fit = np.linspace(min(pos)-0.5, max(pos)+0.5, 800)

            y_fit = beam_model(x_fit, *popt)

            plt.plot(
                x_fit,
                y_fit,
                '--',
                color=colors[i % len(colors)],
                linewidth=2,
                label=f'Fit {label}'
            )

            # ---------------------------------------------------
            # MAXIMUM
            # ---------------------------------------------------
            x_max = x_fit[np.argmax(y_fit)]

            plt.axvline(
                x_max,
                color=colors[i % len(colors)],
                linewidth=1.5
            )

            plt.annotate(
                f'{x_max:.3f}',
                xy=(x_max, 0),
                xycoords=('data', 'axes fraction'),
                xytext=(0, -20 - (i * 18)),
                textcoords='offset points',
                va='top',
                ha='center',
                color=colors[i % len(colors)],
                fontweight='bold'
            )

            # ---------------------------------------------------
            # MIN/MAX FITS MIT x-FEHLER
            # ---------------------------------------------------

            # alle Punkte maximal links
            pos_min = pos - x_error

            # alle Punkte maximal rechts
            pos_max = pos + x_error

            # ---------- MIN FIT ----------
            popt_min, _ = curve_fit(
                beam_model,
                pos_min,
                dose,
                p0=popt,
                bounds=bounds,
                maxfev=50000
            )

            # ---------- MAX FIT ----------
            popt_max, _ = curve_fit(
                beam_model,
                pos_max,
                dose,
                p0=popt,
                bounds=bounds,
                maxfev=50000
            )

            # Fits berechnen
            y_fit_min = beam_model(x_fit, *popt_min)
            y_fit_max = beam_model(x_fit, *popt_max)

            # ---------------------------------------------------
            # ORANGE GRENZFITS
            # ---------------------------------------------------
            plt.plot(
                x_fit,
                y_fit_min,
                ':',
                color=colors[(i+2) % len(colors)],
                linewidth=2,
                alpha=0.9
            )

            plt.plot(
                x_fit,
                y_fit_max,
                ':',
                color=colors[(i+2) % len(colors)],
                linewidth=2,
                alpha=0.9
            )

            # ---------------------------------------------------
            # ORANGE MITTELPUNKTLINIEN
            # ---------------------------------------------------
            x_max_min = x_fit[np.argmax(y_fit_min)]
            x_max_max = x_fit[np.argmax(y_fit_max)]

            plt.axvline(
                x_max_min,
                color=colors[(i+2) % len(colors)],
                linestyle=':',
                linewidth=1.5
            )

            plt.axvline(
                x_max_max,
                color=colors[(i+2) % len(colors)],
                linestyle=':',
                linewidth=1.5
            )

        except Exception as e:
            print(f"Fit fehlgeschlagen ({label}):", e)

    # -------------------------------------------------------
    # PHANTOMPOSITION
    # -------------------------------------------------------
    plt.axvline(
        z_phantom_cm,
        color='darkgreen',
        linewidth=2.5,
        label='Strahlmittelpunkt $S_0$'
    )

    plt.annotate(
        f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f}',
        xy=(z_phantom_cm, 0),
        xycoords=('data', 'axes fraction'),
        xytext=(0, -56),
        textcoords='offset points',
        va='top',
        ha='center',
        color='darkgreen',
        fontweight='bold'
    )
    plt.axvline(kollimiertesStrahlzentrum, color='m', linewidth=2, label='Kollimatormittelpunkt $K_0$')
    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f}', xy=(kollimiertesStrahlzentrum, 0), xycoords=('data', 'axes fraction'), xytext=(0, -70), textcoords='offset points', va='top', ha='center', color='m', fontweight='bold')
    # -------------------------------------------------------
    # LAYOUT
    # -------------------------------------------------------
    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis [mGy]")

    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)

    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.legend()

    plt.subplots_adjust(bottom=0.22)

    plt.show()

def plot_mittelpunkt(datasets, title, suptitle, z_phantom_cm, kollimiertesStrahlzentrum):
    plt.figure(figsize=(10, 6))
    
    # Messwerte
    for data in datasets:
        pos, dose, label, error = data
        
        plt.errorbar(
            pos,
            dose,
            yerr=dose*error,
            label=label,
            fmt='o',
            capsize=5,
            capthick=1.5,
            elinewidth=1.5
        )

    plt.axvline(x=z_phantom_cm, color='darkgreen', linestyle='-', linewidth=2.5, label='Strahlmittelpunkt $S_0$')
    plt.annotate(f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - (len(datasets) * 18)), textcoords='offset points', va='top', ha='right', color='darkgreen', fontweight='bold')
    plt.axvline(kollimiertesStrahlzentrum, color='m', linewidth=2, label='Kollimatormittelpunkt $K_0$')
    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20 - ((len(datasets) + 1) * 18)), textcoords='offset points', va='top', ha='right', color='m', fontweight='bold')

    # Achsen und Beschriftungen
    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis pro Tomographie [mGy]")

    # Titel
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)

    # Legende und Gitter
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.show()


def plot_ptw_plateau(title, suptitle, z_phantom_cm, ptw_avg, ptw_x0=0.0,
                     ptw_length=23.0,      # mm, aktive Kammerlänge
                     ptw_rel_err=0.021,    # 2,1 %
                     ptw_label="PTW",
                     erf_params=None,      # (A, x0, sigma, width) für das schematische Profil
                     x_min=-35.0, x_max=35.0):

    plt.figure(figsize=(10, 6))

    # Rechteck ⊗ Gauß analytisch
    def rect_gauss_erf(x, A, x0, sigma, width):
        return (A/2) * (
            erf((x - x0 + width/2) / (np.sqrt(2)*sigma)) -
            erf((x - x0 - width/2) / (np.sqrt(2)*sigma))
        )

    # -------------------------------------------------
    # schematischer Verlauf des kollimierten Dosisprofils
    # -------------------------------------------------
    if erf_params is not None:
        x_range = np.linspace(x_min, x_max, 500)
        y_schema = rect_gauss_erf(x_range, *erf_params)

        plt.plot(
            x_range,
            y_schema,
            '--',
            color='gray',
            alpha=0.9,
            label='schematisches Dosisprofil'
        )

    # -------------------------------------------------
    # Strahlmittelpunkt und Kollimatormittelpunkt
    # -------------------------------------------------
    plt.axvline(x=z_phantom_cm, color='darkgreen', linestyle='-', linewidth=2.5, label='Strahlmittelpunkt $S_0$')
    plt.annotate(f'Strahlmittelpunkt $S_0$: {z_phantom_cm:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -20), textcoords='offset points', va='top', ha='right', color='darkgreen', fontweight='bold')
    plt.axvline(kollimiertesStrahlzentrum, color='m', linewidth=2, label='Kollimatormittelpunkt $K_0$')
    plt.annotate(f'Kollimatormittelpunkt $K_0$: {kollimiertesStrahlzentrum:.3f} mm', xy=(0.65, 0), xycoords='axes fraction', xytext=(0, -38), textcoords='offset points', va='top', ha='right', color='m', fontweight='bold')

    # -------------------------------------------------
    # PTW-Linie mit Unsicherheit
    # -------------------------------------------------
    ptw_yerr = ptw_avg * ptw_rel_err
    x_left = ptw_x0 - ptw_length / 2
    x_right = ptw_x0 + ptw_length / 2

    # horizontale Linie über 23 mm
    plt.hlines(
        y=ptw_avg,
        xmin=x_left,
        xmax=x_right,
        colors='red',
        linewidth=2.5,
        label=f'{ptw_label}'
    )

    # vertikaler Fehlerbalken in der Mitte
    plt.errorbar(
        ptw_x0,
        ptw_avg,
        yerr=ptw_yerr,
        fmt='none',
        ecolor='red',
        elinewidth=2,
        capsize=6
    )

    # schattiertes Fehlerband über die ganze Länge
    plt.fill_between(
        [x_left, x_right],
        [ptw_avg - ptw_yerr, ptw_avg - ptw_yerr],
        [ptw_avg + ptw_yerr, ptw_avg + ptw_yerr],
        color='red',
        alpha=0.15
    )

    print(f"{ptw_label}:")
    print(f"  Mittelwert = {ptw_avg:.2f} mGy")
    print(f"  Fehler     = ±{ptw_yerr:.2f} mGy")
    print(f"  x-Bereich  = [{x_left:.3f}, {x_right:.3f}] mm")

    plt.xlabel("vertikale Position [mm]", loc='right')
    plt.ylabel("Dosis pro Tomographie [mGy]")
    plt.suptitle(suptitle, fontsize=10, y=0.92)
    plt.title(title, fontsize=16, pad=20)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.subplots_adjust(bottom=0.22)
    plt.show()











def fit_erf(pos, dose):
    def rect_gauss_erf(x, A, x0, sigma, width):
        return (A/2) * (
            erf((x - x0 + width/2) / (np.sqrt(2)*sigma)) -
            erf((x - x0 - width/2) / (np.sqrt(2)*sigma))
        )

    p0 = [
        np.max(dose),          # A
        pos[np.argmax(dose)],  # x0
        5.0,                   # sigma (Kantenbreite)
        30.0                   # width (Feldbreite)
    ]

    bounds = (
        [30, -10.0, 0.5, 10.0],
        [100, 16.0, 20.0, 100.0]
    )

    popt, _ = curve_fit(rect_gauss_erf, pos, dose, p0=p0, bounds=bounds, maxfev=20000)
    return popt


erf_small_center = fit_erf(pos_5, dose_5)
erf_small_off    = fit_erf(pos_6, dose_6)
erf_large_center = fit_erf(pos_4, dose_4)
erf_large_off    = fit_erf(pos_3, dose_3)







# def erf_profile(x, A, x0, sigma, width):
#     return (A/2) * (
#         erf((x - x0 + width/2) / (np.sqrt(2)*sigma)) -
#         erf((x - x0 - width/2) / (np.sqrt(2)*sigma))
#     )


# def integrate_erf(popt, a=-11.5, b=11.5):
#     integral, _ = quad(erf_profile, a, b, args=tuple(popt))
#     mittelwert = integral / (b - a)

#     x_dense = np.linspace(popt[1] - 3*popt[3], popt[1] + 3*popt[3], 20000)
#     maximum = np.max(erf_profile(x_dense, *popt))

#     return integral, mittelwert, maximum

# erf_m1 = fit_erf(pos_1, dose_1)
# erf_m2 = fit_erf(pos_2, dose_2)
# erf_m3 = fit_erf(pos_3, dose_3)
# erf_m4 = fit_erf(pos_4, dose_4)
# erf_m5 = fit_erf(pos_5, dose_5)
# erf_m6 = fit_erf(pos_6, dose_6)

# for name, popts in [("dünn zentral",    [erf_m1, erf_m5]),
#                     ("dünn verschoben", [erf_m2, erf_m6]),
#                     ("dick zentral",    [erf_m4]),
#                     ("dick verschoben", [erf_m3])]:
#     werte = [integrate_erf(p) for p in popts]
#     I  = np.mean([w[0] for w in werte])
#     m  = np.mean([w[1] for w in werte])
#     mx = np.mean([w[2] for w in werte])

#     print(f"{name}:")
#     print(f"  Integral   = {I:.2f} mGy*mm")
#     print(f"  Mittelwert = {m:.3f} mGy")
#     print(f"  Maximum    = {mx:.3f} mGy")
#     print(f"  Mittelwert/Maximum = {m/mx:.4f}  ({100*(1 - m/mx):.2f} % unter dem Maximum)")


#----------------------------------------------------
# Plateau

# plot_ptw_plateau(
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
#     ptw_avg=ptw_small_center_avg,
#     ptw_x0=0.0,
#     ptw_length=23.0,
#     ptw_rel_err=ptw_error,
#     ptw_label="PTW dünn zentral",
#     erf_params=erf_small_center
# )


# plot_ptw_plateau(
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
#     ptw_avg=ptw_small_off_avg,
#     ptw_x0=0.0,
#     ptw_length=23.0,
#     ptw_rel_err=ptw_error,
#     ptw_label="PTW dünn verschoben",
#     erf_params=erf_small_off
# )


# plot_ptw_plateau(
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
#     ptw_avg=ptw_large_center_avg,
#     ptw_x0=0.0,
#     ptw_length=23.0,
#     ptw_rel_err=ptw_error,
#     ptw_label="PTW dick zentral",
#     erf_params=erf_large_center
# )


# plot_ptw_plateau(
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
#     ptw_avg=ptw_large_off_avg,
#     ptw_x0=0.0,
#     ptw_length=23.0,
#     ptw_rel_err=ptw_error,
#     ptw_label="PTW dick verschoben",
#     erf_params=erf_large_off
# )

# # ============================================================
# # ERF-Plots
# # ============================================================


plot_profiles_erf(
    [
        (pos_1, dose_1, "Messung 1", error_123),
        (pos_5, dose_5, "Messung 5", error_456)
    ],
    "Dosisprofil   r=10mm (dünnes) Phantom",
    "Offset: 0mm (zentrale Position)",
    zPhantomebene,
    ptw_avg=ptw_small_center_avg,
    ptw_x0=0.0,
    ptw_length=23.0,
    ptw_rel_err=ptw_error,
    ptw_label="PTW dünn zentral"
)


plot_profiles_erf(
    [
        (pos_2, dose_2, "Messung 2", error_123),
        (pos_6, dose_6, "Messung 6", error_456)
    ],
    "Dosisprofil   r=10mm (dünnes) Phantom",
    "Offset: 10,5mm (Rotationsachsenverschiebung)",
    zPhantomebene,
    ptw_avg=ptw_small_off_avg,
    ptw_x0=0.0,
    ptw_length=23.0,
    ptw_rel_err=ptw_error,
    ptw_label="PTW dünn verschoben"
)


plot_profiles_erf(
    [
        (pos_4, dose_4, "Messung 4", error_456)
    ],
    "Dosisprofil   r=12,5mm (dickes) Phantom",
    "Offset: 0mm (zentrale Position)",
    zPhantomebene,
    ptw_avg=ptw_large_center_avg,
    ptw_x0=0.0,
    ptw_length=23.0,
    ptw_rel_err=ptw_error,
    ptw_label="PTW dick zentral"
)


plot_profiles_erf(
    [
        (pos_3, dose_3, "Messung 3", error_123)
    ],
    "Dosisprofil   r=12,5mm (dickes) Phantom",
    "Offset: 10,5mm (Rotationsachsenverschiebung)",
    zPhantomebene,
    ptw_avg=ptw_large_off_avg,
    ptw_x0=0.0,
    ptw_length=23.0,
    ptw_rel_err=ptw_error,
    ptw_label="PTW dick verschoben"
)



# plot_profiles_erf(
#     [
#         (pos_1, dose_1, "Messung 1", error_123),
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )


# plot_profiles_erf(
#     [
#         (pos_2, dose_2, "Messung 2", error_123),
#         (pos_6, dose_6, "Messung 6", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
# )


# plot_profiles_erf(
#     [
#         (pos_4, dose_4, "Messung 4", error_456)
#     ],
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )


# plot_profiles_erf(
#     [
#         (pos_3, dose_3, "Messung 3", error_123)
#     ],
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
# )

# ============================================================
# mit Fehler erf
# ============================================================

# plot_profiles_erf_mitFehler(
#     [
#         (pos_1, dose_1, "Messung 1", error_123)
#     ],
#     "Dosisprofil   Messung 1",
#     "dünnes Phantom, Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )

# plot_profiles_erf_mitFehler(
#     [
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   Messung 5",
#     "dünnes Phantom, Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )


# plot_profiles_erf_mitFehler(
#     [
#         (pos_2, dose_2, "Messung 2", error_123)
#     ],
#     "Dosisprofil   Messung 2",
#     "dünnes Phantom, Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
# )

# plot_profiles_erf_mitFehler(
#     [
#         (pos_6, dose_6, "Messung 6", error_456)
#     ],
#     "Dosisprofil   Messung 6",
#     "dünnes Phantom, Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
# )


# plot_profiles_erf_mitFehler(
#     [
#         (pos_4, dose_4, "Messung 4", error_456)
#     ],
#     "Dosisprofil   Messung 4",
#     "dickes Phantom, Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )


# plot_profiles_erf_mitFehler(
#     [
#         (pos_3, dose_3, "Messung 3", error_123)
#     ],
#     "Dosisprofil   Messung 3",
#     "dickes Phantom, Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
# )



# ============================================================
# Englisch
# ============================================================

# plot_profiles_erf(
#     [
#         (pos_1, dose_1, "TLD measurement 1", error_123),
#         (pos_5, dose_5, "TLD measurement 5", error_456)
#     ],
#     "Dose profile   r=10mm (thin) phantom",
#     "Offset: 0mm (central position)",
#     zPhantomebene,
#     ptw_avg=ptw_small_center_avg,
#     ptw_x0=0.0,
#     ptw_length=23.0,
#     ptw_rel_err=0.011,
#     ptw_label="PTW thin central"
# )


# plot_profiles_erf(
#     [
#         (pos_4, dose_4, "TLD measurement 4", error_456)
#     ],
#     "Dose profile   r=12.5mm (thick) phantom",
#     "Offset: 0mm (central position)",
#     zPhantomebene,
#     ptw_avg=ptw_large_center_avg,
#     ptw_x0=0.0,
#     ptw_length=23.0,
#     ptw_rel_err=0.011,
#     ptw_label="PTW thick central"
# )



# # # ============================================================
# # # Gauß-Fit und konv fit
# # # ============================================================

# plot_profiles_with_fit(
#     [
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )
# plot_profiles_konvolution(
#     [
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )
# plot_profiles_erf(
#     [
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
# )



# # # ============================================================
# # # Messwerte ohne Fit
# # # ============================================================

# plot_profiles(
#     [
#         (pos_1, dose_1, "Messung 1", error_123),
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
# )


# plot_profiles(
#     [
#         (pos_2, dose_2, "Messung 2", error_123),
#         (pos_6, dose_6, "Messung 6", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
# )


# plot_profiles(
#     [
#         (pos_4, dose_4, "Messung 4", error_456)
#     ],
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 0mm (zentrale Position)",
# )


# plot_profiles(
#     [
#         (pos_3, dose_3, "Messung 3", error_123)
#     ],
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
# )



# # # ============================================================
# # # Gedrehte Darstellung
# # # ============================================================

# plot_profiles_rotated(
#     [
#         (pos_1, dose_1, "Messung 1", error_123),
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
# )


# plot_profiles_rotated(
#     [
#         (pos_2, dose_2, "Messung 2", error_123),
#         (pos_6, dose_6, "Messung 6", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
# )


# plot_profiles_rotated(
#     [
#         (pos_4, dose_4, "Messung 4", error_456)
#     ],
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 0mm (zentrale Position)",
# )


# plot_profiles_rotated(
#     [
#         (pos_3, dose_3, "Messung 3", error_123)
#     ],
#     "Dosisprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
# )


# plot_mittelpunkt(
#     [
#         (pos_1, dose_1, "Messung 1", error_123),
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Dosisprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm (zentrale Position)",
#     zPhantomebene,
#     kollimiertesStrahlzentrum
# )











# plot_profiles_erf_mitFehler(
#     [
#         (pos_1, dose_1, "Messung 1", error_123)
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm",
#     zPhantomebene,
# )
# plot_profiles_erf_mitFehler(
#     [
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm",
#     zPhantomebene,
# )
# plot_profiles_exakt_mitFehler(
#     [
#         (pos_1, dose_1, "Messung 1", error_123)
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm",
#     zPhantomebene,
# )
# plot_profiles_exakt(
#     [
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm",
#     zPhantomebene,
# )
# plot_profiles_exakt(
#     [
#         (pos_1, dose_1, "Messung 1", error_123)
        
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm",
#     zPhantomebene,
# )
# plot_profiles_exakt(
#     [
        
#         (pos_5, dose_5, "Messung 5", error_456)
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 0mm",
#     zPhantomebene,
# )

# plot_profiles_exakt(
#     [
#         (pos_2, dose_2, "Messung 2", error_123)
        
       
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
# )
# plot_profiles_exakt(
#     [
        
#         (pos_6, dose_6, "Messung 6", error_456)
       
#     ],
#     "Strahlprofil   r=10mm (dünnes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene,
# )

# plot_profiles_exakt(
#     [
#         (pos_4, dose_4, "Messung 4", error_456)
#     ],
#     "Strahlprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 0mm",
#     zPhantomebene
# )

# plot_profiles_exakt(
#     [
#         (pos_3, dose_3, "Messung 3", error_123)
#     ],
#     "Strahlprofil   r=12,5mm (dickes) Phantom",
#     "Offset: 10,5mm (Rotationsachsenverschiebung)",
#     zPhantomebene
# )
