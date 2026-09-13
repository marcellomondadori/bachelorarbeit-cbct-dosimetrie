import spekpy as sp # Import SpekPy
import matplotlib.pyplot as plt # Import library for plotting
from scipy.io import loadmat
import numpy as np
import xraydb
from scipy.optimize import brentq
from scipy.integrate import trapezoid

#^ DEFINING FUNCTIONS
#^ ================================================================================
def normalize_area(k, f):
    area = trapezoid(f, k)
    return f / area

def air_kerma_ratio(
    t,
    E,
    phi,
    mu_en_air,
    mu_mat, 
    delta_E
):
    """
    Returns K(t)/K(0) for a given absorber thickness t.
    
    Parameters
    ----------
    t : float
        Absorber thickness (cm)
    E : ndarray
        Photon energies
    phi : ndarray
        Photon fluence per energy bin
    mu_en_air : ndarray
        Mass energy-absorption coefficients for air (cm^2/g)
    mu_mat : ndarray
    Linear attenuation coefficients for absorber (cm^-1)
    rho_mat : float
        Density of absorber (g/cm^3)
    """
    
    attenuation = np.exp(-mu_mat * t)
    
    numerator = np.sum(
        phi * E * mu_en_air * attenuation * delta_E
    )

    denominator = np.sum(
        phi * E * mu_en_air * delta_E
    )

    return numerator / denominator


def compute_HVLs(
    E,
    phi,
    mu_en_air,
    mu_mat,
    delta_E,
    t_max=10.0
):
    """
    Computes first HVL, second HVL, and homogeneity coefficient.
    """
    
    # First HVL
    t1 = brentq(
        lambda t: air_kerma_ratio(t, E, phi, mu_en_air, mu_mat, delta_E) - 0.5,
        0.0,
        t_max
    )

    # Second HVL (total thickness giving 25%)
    t2 = brentq(
        lambda t: air_kerma_ratio(t, E, phi, mu_en_air, mu_mat, delta_E) - 0.25,
        t1,
        t_max
    )

    second_HVL = t2 - t1
    HC = t1 / second_HVL

    return t1, second_HVL, HC

def compute_ef_en(t1, E, mu_mat):
    """
    Computes the effective energy corresponding to the first HVL.

    Parameters
    ----------
    t1 : float
        First HVL in cm.
    E : ndarray
        Photon energies in keV.
    mu_mat : ndarray
        Linear attenuation coefficients of aluminum in cm^-1.
    """

    mu_half = np.log(2) / t1

    eff_E = np.exp(
        np.interp(
            np.log(mu_half),
            np.log(mu_mat[::-1]),
            np.log(E[::-1])
        )
    )

    return eff_E




#^ CALCULATING DIN SPECTRA
#^ ================================================================================
s = sp.Spek(kvp=65,th=20, physics='spekpy-v1', mu_data_source='nist', char=False) # Create a spectrum
s.summarize()
# s.set(ref_kerma=5)
s.filter('Be', 7)
k, f = s.get_spectrum(edges=True) # Get the spectrum
f_n = normalize_area(k, f)
s.filter('Al', 1.56) # Filter the spectrum
k_filtered, f_filtered = s.get_spectrum(edges=True) # Get the spectrum
f_filtered_n = normalize_area(k_filtered, f_filtered)
#^ ================================================================================



# #^ CALCULATING CBCT System SPECTRA
# #^ ================================================================================
s2 = sp.Spek(kvp=65,th=12, physics='spekpy-v1', mu_data_source='nist', char=False)
# s2.set(ref_kerma=20)
s2.filter('Be', 0.1)
k2, f2 = s2.get_spectrum(edges=True) # Get the spectrum
f2_n = normalize_area(k2, f2)
s2.filter('Cu', 0.1)



s2.summarize()

hvl_al_mm = s2.get_hvl1(matl='Al', to='air')
eeff_al_kev = s2.get_eeff(matl='Al', to='air')

print(f"SpekPy-HVL1 Al = {hvl_al_mm:.6f} mm")
print(f"SpekPy-Eeff Al = {eeff_al_kev:.6f} keV")




k2_filtered, f2_filtered = s2.get_spectrum(edges=True) # Get the spectrum
f2_filtered_n = normalize_area(k2_filtered, f2_filtered)
# #^ ================================================================================

#^ CALCULATING CBCT System SPECTRA
#^ ================================================================================
# s2 = sp.Spek(kvp=65, th=12, physics='spekpy-v1', mu_data_source='nist', char=False)

# # 1. Spektrum nach dem Röhrenfilter (0.1 mm Kupfer)
# s2.filter('Cu', 0.1) 
# k2, f2 = s2.get_spectrum(edges=True) 
# f2_n = normalize_area(k2, f2) # Das ist dein "unfiltered" (nur Röhre)

# 2. Spektrum nach dem Phantom (20 mm PMMA)
# # Wir berechnen die effektive Dicke für jedes Element basierend auf den Massenanteilen
# # PMMA Dichte: 1.19 g/cm3 | Dicke: 20 mm
# d_pmma = 12.5  # mm
# rho_pmma = 1.19

# # Wir filtern nacheinander mit den Elementen (SpekPy kennt 'H', 'C', 'O' als Standard)
# # Dicke pro Element = Gesamt-Dicke * Massenanteil
# s2.filter('H', d_pmma * 0.0805)
# s2.filter('C', d_pmma * 0.5998)
# s2.filter('O', d_pmma * 0.3196)

# k2_filtered, f2_filtered = s2.get_spectrum(edges=True) 
# f2_filtered_n = normalize_area(k2_filtered, f2_filtered)

# #^ ================================================================================




#^ LOADING MEASURED SPECTRUM
#^ ================================================================================
from pathlib import Path

data_path = str(Path(__file__).resolve().parents[1] / "daten") + "/"

data = loadmat(data_path + "65kVp_measurement_interp.mat")
k_meas = data['spectrum_x'][0]
f_meas_n = normalize_area(k_meas, data['spectrum_y'][0])
#^ ================================================================================


#^ PLOTTING
#^ ================================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))

# Vergleich von Simulation und Messung ohne zusätzliche Cu-Filterung
ax.plot(
    k2,
    f2_n,
    color='#006400',  # dunkelgrün
    linestyle='-',
    linewidth=2,
    label='Simulation μCBCT ohne Cu-Filter: 0,1 mm Be'
)

ax.plot(
    k_meas,
    f_meas_n,
    color='#66BB6A',  # hellgrün
    linestyle='--',
    linewidth=2,
    label='Messung μCBCT ohne Cu-Filter: 0,1 mm Be'
)

# Vergleich des gefilterten CBCT-Spektrums mit der DIN-Referenzqualität
ax.plot(
    k2_filtered,
    f2_filtered_n,
    color='tab:red',
    linewidth=2,
    label='Simulation μCBCT mit Cu-Filter: 0,1 mm Be + 0,1 mm Cu'
)

ax.plot(
    k_filtered,
    f_filtered_n,
    color='tab:orange',
    linestyle='--',
    linewidth=2,
    label='DIN-Referenzqualität: 7 mm Be + 1,56 mm Al'
)

ax.set_xlabel('Photonenenergie [keV]')
ax.set_ylabel(r'Normierte spektrale Photonenfluenz [keV$^{-1}$]')

ax.set_xlim(0, 65)
ax.set_ylim(bottom=0)

ax.grid(alpha=0.25)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.22),
    ncol=2,
    fontsize=9
)

fig.tight_layout()

fig.savefig(
    'Spektrenvergleich.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
#^ ================================================================================


# #^ CALCULATING HVL & EFFECTIVE ENERGY
# #^ 1. 
# #^ ================================================================================
# E = k_meas_filt
# print(f"length E: {len(E)}")
# print(f"E: \n{E}")
# phi = f_meas_filt_n
# print(f"length phi: {len(phi)}")
# print(f"phi: {phi}")
# mu_en_air = np.array([xraydb.material_mu('Air', float(Ei*1000)) for Ei in E])  # cm^2/g
# mu_PMMA = np.array([xraydb.material_mu('PMMA', float(Ei*1000)) for Ei in E])       # cm^2/g
# # rho_Al = 2.70                                     # g/cm^3
# delta_E = np.ones_like(E)
# delta_E[:] = 0.5
# print(f"length delta_E: {len(delta_E)}")
# print(f"delta_E: {delta_E}")

# t1, t2, HC = compute_HVLs(E, phi, mu_en_air, mu_PMMA, delta_E)
# eff_E = compute_ef_en(t1, E)

# print(f"Half value material: PMMA")
# print(f"First HVL  = {t1:.3f} cm")
# print(f"Second HVL = {t2:.3f} cm")
# print(f"HC         = {HC:.3f}")
# print(f"effective energy = {eff_E:.3f}")
#^ ================================================================================

#^ 2. HVL AND EFFECTIVE ENERGY OF THE CBCT SPECTRUM
#^ ================================================================================

# Mid-bin energies and corresponding photon fluence
E, phi = s2.get_spectrum(edges=False)
phi = normalize_area(E, phi)

# Mass energy-absorption coefficients of air in cm^2/g
mu_en_air = s2.muen_air_data.get_muen_over_rho_air(E)

# Linear attenuation coefficients of aluminum in cm^-1
mu_Al = s2.mu_data.get_mu_composition('Al', E)

# Energy-bin width in keV
delta_E = np.full_like(E, E[1] - E[0])

t1, t2, HC = compute_HVLs(
    E,
    phi,
    mu_en_air,
    mu_Al,
    delta_E
)

eff_E = compute_ef_en(t1, E, mu_Al)

print("Half value material: Al")
print(f"First HVL  = {t1 * 10:.6f} mm")
print(f"Second HVL = {t2 * 10:.6f} mm")
print(f"HC         = {HC:.6f}")
print(f"Effective energy = {eff_E:.6f} keV")

#^ ================================================================================



mean_energy = s2.get_emean()
hvl_al = s2.get_hvl1(matl='Al', to='air')
effective_energy = s2.get_eeff(matl='Al', to='air')

print(f"Mean energy = {mean_energy:.6f} keV")
print(f"HVL1 Al = {hvl_al:.6f} mm")
print(f"Effective energy = {effective_energy:.6f} keV")













#^ 3. WECHSELWIRKUNGSANTEILE IN PMMA (NIST XCOM) MIT SPEKTRUM IM HINTERGRUND
#^ ================================================================================
from matplotlib.patches import Patch

# XCOM-Ausgabe für Compound C5H8O2 als Textdatei im Ordner data_path
rows = []
with open(data_path + "xcom_pmma.txt", encoding="utf-8") as file:
    for line in file:
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            continue                      # Kopfzeilen und Z=...-Zeilen überspringen
        if len(vals) == 8:
            rows.append(vals)
xcom = np.array(rows)
xcom = xcom[(xcom[:, 0] >= 0.010) & (xcom[:, 0] <= 0.100)]

E_x = xcom[:, 0] * 1e3                    # MeV -> keV
coh, incoh, photo, total = xcom[:, 1], xcom[:, 2], xcom[:, 3], xcom[:, 6]


def loglog(e, y):
    """Doppelt-logarithmische Interpolation der XCOM-Tabellenwerte"""
    return np.exp(np.interp(np.log(e), np.log(E_x), np.log(y)))


# Kennzahlen für den Text
E_cross = brentq(lambda e: np.log(loglog(e, photo) / loglog(e, incoh)), 15, 40)
frac_below = 100 * np.sum(phi[E < E_cross]) / np.sum(phi)   # E, phi: gefiltertes Spektrum aus 2.

print(f"Schnittpunkt Photoeffekt/Compton = {E_cross:.1f} keV")
print(f"Photonenanteil unterhalb         = {frac_below:.0f} %")
for e in (eff_E, 65.0):
    print(f"Rayleigh-Anteil bei {e:4.1f} keV     = {100 * loglog(e, coh) / loglog(e, total):.1f} %")

# Plot
E_fine = np.linspace(10, 70, 400)
processes = [
    (photo, 'Photoeffekt', 'tab:blue'),
    (incoh, 'Compton-Streuung', 'tab:orange'),
    (coh, 'Rayleigh-Streuung', 'tab:green'),
]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 8))

for y, label, color in processes:
    ax1.semilogy(E_fine, loglog(E_fine, y), color=color, linewidth=2, label=label)
    ax1.semilogy(E_x, y, 'o', color=color, markersize=4)
    ax2.plot(E_fine, 100 * loglog(E_fine, y) / loglog(E_fine, total), color=color, linewidth=2)
ax1.semilogy(E_fine, loglog(E_fine, total), 'k--', linewidth=1.5, label='Gesamt')

for ax in (ax1, ax2):
    ax.axvline(eff_E, color='grey', linestyle=':', label=r'$E_{\mathrm{eff}}$')
    ax.axvline(65, color='grey', linestyle='-.', label=r'$E_{\mathrm{max}}$')
    ax.grid(alpha=0.25)

# Gefiltertes Spektrum als graue Fläche im unteren Panel
ax2_spec = ax2.twinx()
ax2_spec.fill_between(E, phi / phi.max(), color='grey', alpha=0.25, linewidth=0)
ax2_spec.set_ylim(0, 1.05)
ax2_spec.set_yticks([])
ax2_spec.set_ylabel('Spektrum (normiert)', color='grey')
ax2.set_zorder(ax2_spec.get_zorder() + 1)   # Kurven vor die Fläche legen
ax2.patch.set_visible(False)

handles, labels = ax1.get_legend_handles_labels()
handles.append(Patch(color='grey', alpha=0.25))
labels.append('Spektrum (unten)')
ax1.legend(handles, labels, fontsize=9, handlelength=3)

ax1.set_ylabel(r'$\mu/\rho$ [cm$^2$/g]')
ax2.set_ylabel('Anteil an der Gesamtschwächung [%]')
ax2.set_xlabel('Photonenenergie [keV]')
ax2.set_xlim(10, 70)
ax2.set_ylim(0, 100)

fig.tight_layout()
fig.savefig('XCOM-PMMA.png', dpi=300, bbox_inches='tight')
plt.show()
#^ ================================================================================




import spekpy as sp
for x in (-1.83, 1.83):  # cm entlang Anoden-Kathoden-Achse
    s = sp.Spek(kvp=65, th=12, dk=0.5, physics='spekpy-v1', x=x, z=44.1)
    s.filter('Be', 0.1)
    s.filter('Cu', 0.1)
    print(x, s.get_kerma())