"""
kQ-Bestimmung aus effektiver Energie bzw. Vergleichshalbwertsdicke.

Ablauf: erst Eingabe, dann Plot mit rot markiertem, interpoliertem Punkt.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# =====================================================================
#  HILFSFUNKTIONEN
# =====================================================================
def de(x, nd=3):
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def _to_axes_frac(ax, x, y):
    """Datenkoordinaten -> Achsenbruchteile (0..1)."""
    pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
    return ax.transAxes.inverted().transform(ax.transData.transform(pts))


def _legend_box_frac(ax, leg):
    """Legendenrechteck in Achsenbruchteilen, oder None."""
    if leg is None:
        return None
    ax.figure.canvas.draw()          # noetig, sonst hat die Legende keine Groesse
    bb = leg.get_window_extent()
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(
        [[bb.x0, bb.y0], [bb.x1, bb.y1]]
    )
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _dist_to_rect(px, py, rect):
    """Abstand eines Punktes zu einem Rechteck (0 wenn innerhalb)."""
    if rect is None:
        return np.inf
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return np.hypot(dx, dy)


def annotate_point(ax, x_pt, y_pt, text, curves, leg=None,
                   margin=0.10, prefer_near=0.30):
    """
    Setzt einen Beschriftungskasten an die freieste Stelle der Achse und
    verbindet ihn mit einer Linie zum markierten Punkt.

    curves : Liste von (x, y)-Arrays, denen der Kasten ausweichen soll
    leg    : Legendenobjekt, dem der Kasten ausweichen soll
    """
    # Alles, was im Weg ist, in Achsenbruchteile umrechnen
    obstacles = []
    for cx, cy in curves:
        obstacles.append(_to_axes_frac(ax, cx, cy))
    obstacles.append(_to_axes_frac(ax, x_pt, y_pt))
    obstacles = np.vstack(obstacles)

    rect = _legend_box_frac(ax, leg)
    p = _to_axes_frac(ax, x_pt, y_pt)[0]

    # Kandidatenraster, Rand ausgespart
    gx, gy = np.meshgrid(np.linspace(0.14, 0.86, 13),
                         np.linspace(0.14, 0.86, 13))
    best, best_score = None, -np.inf

    for cx, cy in zip(gx.ravel(), gy.ravel()):
        d_obs = np.min(np.hypot(obstacles[:, 0] - cx, obstacles[:, 1] - cy))
        d_leg = _dist_to_rect(cx, cy, rect)
        d_pt = np.hypot(p[0] - cx, p[1] - cy)

        # zu nah an Kurve/Legende oder direkt auf dem Punkt: unbrauchbar
        if d_obs < margin or d_leg < margin or d_pt < 0.18:
            continue

        # frei stehen ist wichtig, kurze Verbindungslinie ist ein Bonus
        score = min(d_obs, d_leg) - 0.25 * abs(d_pt - prefer_near)
        if score > best_score:
            best_score, best = score, (cx, cy)

    if best is None:                 # Notfall: unten rechts
        best = (0.72, 0.18)

    ax.annotate(
        text,
        xy=(x_pt, y_pt), xycoords="data",
        xytext=best, textcoords="axes fraction",
        ha="center", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="crimson", linewidth=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="crimson", linewidth=1.1,
                        shrinkA=2, shrinkB=6,
                        connectionstyle="arc3,rad=0.0"),
        zorder=6,
    )


def mark_point(ax, x_pt, y_pt):
    """Roter Punkt mit gestrichelten Hilfslinien zu beiden Achsen."""
    x0 = ax.get_xlim()[0]
    y0 = ax.get_ylim()[0]
    ax.plot([x0, x_pt], [y_pt, y_pt], ls=":", color="crimson",
            linewidth=1.0, zorder=4)
    ax.plot([x_pt, x_pt], [y0, y_pt], ls=":", color="crimson",
            linewidth=1.0, zorder=4)
    ax.plot([x_pt], [y_pt], "o", color="crimson", markersize=8,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5,
            label="interpolierter Wert")


def ask_float(prompt):
    """Eingabe mit Komma oder Punkt, fragt bei Unsinn nochmal."""
    while True:
        try:
            return float(input(prompt).replace(",", ".").strip())
        except ValueError:
            print("  Bitte eine Zahl eingeben, z. B. 30,15")


# =====================================================================
#  PHANTOM 1: kQ aus effektiver Energie (TLD)
# =====================================================================
Eeff = np.array([11.5, 15.5, 19.8, 22.4, 26.9, 33.5, 42.1, 49.9, 67.0, 99.8, 145.0])
kQ = np.array([0.986, 0.804, 0.720, 0.699, 0.664, 0.668, 0.697, 0.724, 0.769, 0.853, 0.887])

# PCHIP statt CubicSpline: der kubische Spline schwingt am Minimum ueber und
# liefert um 29 keV kQ = 0.6585, also UNTER dem kleinsten Tabellenwert 0.664.
# Das ist ein reines Interpolationsartefakt. PCHIP ist formerhaltend.
interp_E = PchipInterpolator(Eeff, kQ, extrapolate=False)


def get_kQ(E_input):
    val = interp_E(E_input)
    if np.isnan(val):
        print("  Warnung: ausserhalb des Tabellenbereichs, keine Interpolation moeglich.")
    return float(val)


print("=" * 60)
print("Phantom 1 (TLD): kQ aus effektiver Energie")
print(f"Tabellenbereich: {de(Eeff[0],1)} bis {de(Eeff[-1],1)} keV")
print("=" * 60)

E_test = ask_float("Gib einen Eeff-Wert [keV] ein: ")
kQ_result = get_kQ(E_test)
print(f"\nFuer Eeff = {de(E_test,2)} keV ist kQ = {de(kQ_result,4)}\n")

fig, ax = plt.subplots(figsize=(8, 5))
E_fine = np.linspace(Eeff[0], Eeff[-1], 500)
ax.plot(Eeff, kQ, "o", color="tab:blue", label="Tabellenwerte")
ax.plot(E_fine, interp_E(E_fine), "-", color="tab:blue",
        linewidth=1.5, label="PCHIP-Interpolation")

if not np.isnan(kQ_result):
    mark_point(ax, E_test, kQ_result)

ax.set_xlabel("effektive Energie $E_{eff}$ [keV]")
ax.set_ylabel("$k_Q$")
ax.set_title("TLD: $k_Q$ aus effektiver Energie")
ax.grid(True, alpha=0.35)
leg = ax.legend(loc="upper right")

if not np.isnan(kQ_result):
    annotate_point(
        ax, E_test, kQ_result,
        f"$E_{{eff}}$ = {de(E_test, 2)} keV\n$k_Q$ = {de(kQ_result, 4)}",
        curves=[(E_fine, interp_E(E_fine)), (Eeff, kQ)],
        leg=leg,
    )

fig.tight_layout()
plt.show()


# =====================================================================
#  PHANTOM 2: kQ aus Al-Halbwertsdicke (PTW Farmer 30013)
# =====================================================================
HVL_Al = np.array([0.11, 0.44, 1.13, 3.15])
kQ_PTW = np.array([1.272, 1.000, 0.925, 0.896])
NAMES = ["TW15", "TW30", "TW50", "TW70"]

pchip = PchipInterpolator(HVL_Al, kQ_PTW, extrapolate=False)


def get_kQ_from_HVL(hvl):
    """PCHIP innerhalb des Kalibrierbereichs, linear extrapoliert ausserhalb."""
    hvl = np.atleast_1d(np.asarray(hvl, dtype=float))
    result = np.empty_like(hvl)

    inside = (hvl >= HVL_Al[0]) & (hvl <= HVL_Al[-1])
    result[inside] = pchip(hvl[inside])

    slope_left = (kQ_PTW[1] - kQ_PTW[0]) / (HVL_Al[1] - HVL_Al[0])
    left = hvl < HVL_Al[0]
    result[left] = kQ_PTW[0] + slope_left * (hvl[left] - HVL_Al[0])

    slope_right = (kQ_PTW[-1] - kQ_PTW[-2]) / (HVL_Al[-1] - HVL_Al[-2])
    right = hvl > HVL_Al[-1]
    result[right] = kQ_PTW[-1] + slope_right * (hvl[right] - HVL_Al[-1])

    return float(result[0]) if result.size == 1 else result


print("=" * 60)
print("Phantom 2 (PTW Farmer 30013): kQ aus Al-Halbwertsdicke")
print(f"Kalibrierbereich: {de(HVL_Al[0],2)} bis {de(HVL_Al[-1],2)} mm Al")
print("=" * 60)

HVL_input = ask_float("Gib die Vergleichshalbwertsdicke in Al [mm] ein: ")
kQ_hvl = get_kQ_from_HVL(HVL_input)
extrapolated = not (HVL_Al[0] <= HVL_input <= HVL_Al[-1])

if extrapolated:
    print("  Achtung: ausserhalb des Kalibrierbereichs, linear extrapoliert.")
print(f"\nFuer HVL = {de(HVL_input,3)} mm Al ist kQ = {de(kQ_hvl,4)}\n")

fig, ax = plt.subplots(figsize=(8, 5))
HVL_fine = np.linspace(min(0.05, HVL_input * 0.9),
                       max(3.60, HVL_input * 1.1), 500)
ax.plot(HVL_fine, get_kQ_from_HVL(HVL_fine), "-", color="tab:blue",
        linewidth=1.5, label="PCHIP mit linearer Extrapolation")
ax.plot(HVL_Al, kQ_PTW, "o", color="tab:blue", label="PTW-Kalibrierpunkte")

for h, q, name in zip(HVL_Al, kQ_PTW, NAMES):
    ax.annotate(name, (h, q), xytext=(6, 6), textcoords="offset points",
                fontsize=9, color="dimgray")


mark_point(ax, HVL_input, kQ_hvl)

ax.set_xlabel("H2albwertsdicke in Al [mm]")
ax.set_ylabel("$k_Q$")
ax.set_title("PTW Farmer Chamber TM 30013: $k_Q$ aus Al-Halbwertsdicke")
ax.grid(True, alpha=0.35)
leg = ax.legend(loc="upper right")

label = f"HVL = {de(HVL_input, 3)} mm Al\n$k_Q$ = {de(kQ_hvl, 4)}"
if extrapolated:
    label += "\n(extrapoliert)"

annotate_point(
    ax, HVL_input, kQ_hvl, label,
    curves=[(HVL_fine, get_kQ_from_HVL(HVL_fine)), (HVL_Al, kQ_PTW)],
    leg=leg,
)

fig.tight_layout()
plt.show()