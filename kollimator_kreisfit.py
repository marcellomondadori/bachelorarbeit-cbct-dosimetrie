from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from pathlib import Path

file_path = Path(__file__).resolve().parents[1] / "daten" / "beamradius_TE.EVI"

width = 512
height = 256
offset = 265408
dtype = np.dtype("<u2")

# =========================
# Bild laden
# =========================

with open(file_path, "rb") as f:
    f.seek(offset)
    img = np.fromfile(f, dtype=dtype, count=width*height)
    img = img.reshape((height, width))

# nur linke Hälfte verwenden
img = img[:, :256]

height, width = img.shape

# =========================
# Kantenpunkte finden
# =========================

x_points = []
y_points = []

# linker Kreisbogen
for x in range(0, 8):

    column = img[232:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 232 + idx

    x_points.append(x)
    y_points.append(y)

for x in range(8, 19):

    column = img[239:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 239 + idx

    x_points.append(x)
    y_points.append(y)

for x in range(19, 21):

    column = img[248:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 248 + idx

    x_points.append(x)
    y_points.append(y)

# rechter Kreisbogen
for x in range(235, 240):

    column = img[232:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 232 + idx

    x_points.append(x)
    y_points.append(y)

for x in range(240, 241):

    column = img[247:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 247 + idx

    x_points.append(x)
    y_points.append(y)

for x in range(241, 251):

    column = img[232:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 232 + idx

    x_points.append(x)
    y_points.append(y)

for x in range(251, 252):

    column = img[240:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 240 + idx

    x_points.append(x)
    y_points.append(y)

for x in range(252, 256):

    column = img[232:, x].astype(float)

    grad = np.abs(np.diff(column))

    idx = np.argmax(grad)

    y = 232 + idx

    x_points.append(x)
    y_points.append(y)

x_points = np.array(x_points)
y_points = np.array(y_points)

# =========================
# Kreisfit
# =========================

def circle_residuals(params, x, y):
    a, b, r = params
    return np.sqrt((x-a)**2 + (y-b)**2) - r

# Startwerte nahe Erwartung
a0 = 128
b0 = 115
r0 = 178

result = least_squares(
    circle_residuals,
    x0=[a0, b0, r0],
    args=(x_points, y_points)
)

a_fit, b_fit, r_fit = result.x

# =========================
# Fehler bestimmen (Bootstrap)
# =========================

n_boot = 1000

a_boot = []
b_boot = []
r_boot = []

rng = np.random.default_rng()

n_subset = int(len(x_points) * 2 / 3)

for _ in range(n_boot):

    idx = rng.choice(
        len(x_points),
        size=n_subset,
        replace=False
    )

    x_sub = x_points[idx]
    y_sub = y_points[idx]

    try:

        res = least_squares(
            circle_residuals,
            x0=[a_fit, b_fit, r_fit],
            args=(x_sub, y_sub)
        )

        a_b, b_b, r_b = res.x

        a_boot.append(a_b)
        b_boot.append(b_b)
        r_boot.append(r_b)

    except:
        pass

a_boot = np.array(a_boot)
b_boot = np.array(b_boot)
r_boot = np.array(r_boot)


# =========================
# Fehlerband aus Bootstrap
# =========================

x_boot = []
y_boot = []

theta = np.linspace(0, 2*np.pi, 360)

for a, b, r in zip(a_boot, b_boot, r_boot):

    x_boot.append(
        a + r*np.cos(theta)
    )

    y_boot.append(
        b + r*np.sin(theta)
    )

x_boot = np.array(x_boot)
y_boot = np.array(y_boot)

# 16%- und 84%-Grenze

x_low_band = np.percentile(x_boot, 16, axis=0)
x_up_band  = np.percentile(x_boot, 84, axis=0)

y_low_band = np.percentile(y_boot, 16, axis=0)
y_up_band  = np.percentile(y_boot, 84, axis=0)




sigma_a = np.std(a_boot)
sigma_b = np.std(b_boot)
sigma_r = np.std(r_boot)

a_low, a_up = np.percentile(a_boot, [16, 84])
b_low, b_up = np.percentile(b_boot, [16, 84])
r_low, r_up = np.percentile(r_boot, [16, 84])

# =========================
# Kreise erzeugen
# =========================

theta = np.linspace(0, 2*np.pi, 1000)

def circle(a, b, radius):
    x = a + radius*np.cos(theta)
    y = b + radius*np.sin(theta)
    return x, y


x_mid, y_mid = circle(a_fit, b_fit, r_fit)



# =========================
# Plot Kantenpunkte
plt.figure(figsize=(8,8))

plt.imshow(img, cmap="gray", aspect="equal")

plt.scatter(
    x_points,
    y_points,
    s=1,
    label="erkannte Kantenpunkte"
)
plt.legend()

plt.title("Punkte")
plt.show()

# =========================

plt.figure(figsize=(8,8))

plt.imshow(img, cmap="gray", aspect="equal")

plt.scatter(
    x_points,
    y_points,
    s=1,
    label="erkannte Kantenpunkte"
)

plt.plot(
    x_mid,
    y_mid,
    linewidth=1,
    label=f"Fit: R={r_fit:.2f}"
)

plt.plot(
    x_low_band,
    y_low_band,
    "--",
    linewidth=0.8
)

plt.plot(
    x_up_band,
    y_up_band,
    "--",
    linewidth=0.8
)

plt.fill(
    np.concatenate([
        x_up_band,
        x_low_band[::-1]
    ]),
    np.concatenate([
        y_up_band,
        y_low_band[::-1]
    ]),
    alpha=0.2,
    # label="1σ Bootstrap-Band"
)


# =========================
# Mittelpunkte kennzeichnen
# =========================

# Detektormittelpunkt: geometrische Mitte des Bildausschnitts
x_D = (width  - 1) / 2          # = 127.5
y_D = (height - 1) / 2          # = 127.5

# Kreismittelpunkt aus dem Fit
x_K = a_fit
y_K = b_fit

# Höhenlinien
plt.axhline(y_K, color='#6A1B9A', linestyle='--', linewidth=0.8, zorder=4)
plt.axhline(y_D, color='black',   linestyle=':',  linewidth=1.0, zorder=4)

# Marker
plt.plot(x_K, y_K, marker='+', markersize=15, markeredgewidth=2.2,
         color='#6A1B9A', linestyle='none', zorder=5,
         label=rf"Kreismittelpunkt $y_\mathrm{{K}}$ = {y_K:.3f} px")

plt.plot(x_D, y_D, marker='x', markersize=12, markeredgewidth=2.2,
         color='black', linestyle='none', zorder=5,
         label=rf"Detektormittelpunkt $y_\mathrm{{D}}$ = {y_D:.3f} px")
# Abstand der beiden Mitten
plt.annotate("", xy=(x_D, y_D), xytext=(x_D, y_K),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.3), zorder=6)

plt.text(x_D + 8, (y_K + y_D)/2,
         rf"$\Delta y$ = {y_D - y_K:.3f} px",
         va='center', ha='left', color='black', fontweight='bold', zorder=6,
         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

plt.legend(loc='lower right', fontsize=9)   # <- dein bestehender legend-Aufruf


plt.title("Kollimator-Kreisfit")
plt.show()

# =========================
# Ausgabe
# =========================

print(f"Kreiszentrum x = {a_fit:.3f}")
print(f"Kreiszentrum y = {b_fit:.3f}")
print(f"Radius         = {r_fit:.3f}")

print()
print("Bootstrap-Fehler:")

print(f"sigma(x) = {sigma_a:.3f}")
print(f"sigma(y) = {sigma_b:.3f}")
print(f"sigma(R) = {sigma_r:.3f}")

print()
print("16% Unterer Kreis:")
print(f"x = {a_low:.3f}")
print(f"y = {b_low:.3f}")
print(f"R = {r_low:.3f}")

print()
print("84% Oberer Kreis:")
print(f"x = {a_up:.3f}")
print(f"y = {b_up:.3f}")
print(f"R = {r_up:.3f}")

print()
print("Mittelpunkte:")
print(f"y_K (Kreismittelpunkt)     = {b_fit:.3f} px")
print(f"y_D (Detektormittelpunkt)  = {(height - 1) / 2:.3f} px")
print(f"Delta y = y_D - y_K        = {(height - 1) / 2 - b_fit:.3f} px")









# from matplotlib import pyplot as plt
# import numpy as np
# from scipy.optimize import least_squares

# file_path = "/home/m/M.Mondadori/Desktop/Bachelorarbeit/04.17_Strahlradius/beamradius_TE.EVI"

# width = 512
# height = 256
# offset = 265408
# dtype = np.dtype("<u2")

# # =========================
# # Bild laden
# # =========================
# with open(file_path, "rb") as f:
#     f.seek(offset)
#     img = np.fromfile(f, dtype=dtype, count=width*height)
#     img = img.reshape((height, width))

# # =========================
# # 1. Bild: nur Bild
# # =========================
# plt.figure(figsize=(10,5))
# plt.imshow(img, cmap="gray", aspect="equal")
# plt.axis("off")
# plt.title("Bild 2")
# plt.show()


# # =========================
# # Kreis-Fit
# # =========================
# x = np.array([0.6, 16.2, 5.3, 236.9, 255.7])
# y = np.array([239.2, 254.9, 243.8, 255.1, 240.2])

# def circle_residuals(params, x, y):
#     a, b, r = params
#     return (x - a)**2 + (y - b)**2 - r**2

# x_m = np.mean(x)
# y_m = np.mean(y)
# r0 = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))

# result = least_squares(circle_residuals, x0=[x_m, y_m, r0], args=(x, y))
# a_fit, b_fit, r_fit = result.x

# # Kreispunkte
# theta = np.linspace(0, 2*np.pi, 300)
# x_circle = a_fit + r_fit * np.cos(theta)
# y_circle = b_fit + r_fit * np.sin(theta)

# # =========================
# # 2. Bild: Bild + Kreis
# # =========================
# plt.figure(figsize=(10,5))

# plt.imshow(img, cmap="gray", aspect="equal")

# # Kreis drüber plotten
# plt.plot(x_circle, y_circle, linewidth=2)

# # optional: Punkte anzeigen
# plt.scatter(x, y, s=10)

# plt.axis("off")
# plt.title("Bild 2 mit Kreis-Fit")
# plt.show()

# # =========================
# # Output
# # =========================
# print(f"Kreiszentrum: ({a_fit:.2f}, {b_fit:.2f})")
# print(f"Radius: {r_fit:.2f}")