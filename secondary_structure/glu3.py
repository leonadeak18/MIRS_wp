# Deconvolution of FTIR data
import ftir_secondary_structure as ss
import pandas as pd
import numpy as np
from scipy.special import voigt_profile
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# Load the FTIR data
data = pd.read_csv("data/glu3.csv", header=0)
data.rename(columns={data.columns[0]: "wavenumber"}, inplace=True)

# Preprocess the data
pre_data = ss.preprocessing(data, window=7, normalize=False)
valley_picking = ss.valley_peaks(pre_data)
processed_data = ss.baseline_correction(pre_data, valley_picking.drop(valley_picking.index[1], axis=0))

# Calculate the total area under the curve between 1600 and 1700
wavenumber_zero = processed_data[processed_data["baseline_corrected"] == 0]
w1 = wavenumber_zero["wavenumber"].min()
w2 = wavenumber_zero["wavenumber"].max()
area = ss.total_area(processed_data, w1=w1, w2=w2)

# deconvoluted the FTIR peaks
xy = processed_data[(processed_data["wavenumber"] <= w2) & (processed_data["wavenumber"] >= w1)]
x = xy["wavenumber"]
y = xy["baseline_corrected"]

#plotting the FTIR data
plt.figure(figsize=(7, 5))
plt.plot(x, y, "bo", markersize=3, label="Experimental Data")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorbance (a.u.)")
plt.gca().invert_xaxis()

## Provide initial parameter guesses for 5 peaks
initial_params = [
    #Amplitude, Center, Sigma, Gamma
    #0.00005, 1690, 0, 0, # Peak 1
    0.0002, 1675, 7, 3, # Peak 2
    0.0003, 1667, 7, 3, # Peak 3
    0.0002, 1656, 7, 3, # Peak 4
    0.0002, 1648, 7, 3, # Peak 5
    0.00002, 1638, 1, 1, # Peak 7
    0.0001, 1628, 7, 3, # Peak 6
    0.00006, 1613, 10, 3, # Peak 8
]

lower_bounds = [
    #Amplitude, Center, Sigma, Gamma
    0, initial_params[1]- 1, 0, 0,  # Peak1 
    0, initial_params[5] - 1, 0, 0,  # Peak2
    0, initial_params[9] - 2, 0, 0,  # Peak3
    0, initial_params[13] - 2, 0, 0,  # Peak4
    0, initial_params[17] - 2, 0, 0,  # Peak5
    0, initial_params[21] - 2, 0, 0,  # Peak6
    0, initial_params[25] - 2, 0, 0,  # Peak7
]
upper_bounds = [
    #Amplitude, Center, Sigma, Gamma
    np.inf, initial_params[1] + 1, 20, 20,  # Peak1
    np.inf, initial_params[5] + 1, 20, 20,  # Peak2
    np.inf, initial_params[9] + 2, 20, 20,  # Peak3
    np.inf, initial_params[13] + 2, 20, 20,  # Peak4
    np.inf, initial_params[17] + 2, 20, 20,  # Peak5
    np.inf, initial_params[21] + 2, 20, 20,  # Peak6
    np.inf, initial_params[25] + 2, 20, 20,  # Peak7
]
glu3 = ss.deconvolution(x, y, initial_params, lower_bounds, upper_bounds)

# Plot Results
plt.figure(figsize=(7, 5))
plt.plot(x, y, "bo", markersize=3, label="Experimental Data")
plt.plot(x, glu3["y_fit"], "r-", linewidth=2, label="Total Fit")
colors = [ "g", "m", "c", "y", "k", "r", "b"]
for i in range(7):
    plt.plot(
        x,
        glu3["amplitudes"][i] * voigt_profile(x - glu3["centers"][i], glu3["sigmas"][i], glu3["gammas"][i]),
        colors[i],
        linestyle="--",
        label=f"Peak {i + 1}",
    )
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorbance (a.u.)")
plt.legend()
plt.title(f"adjusted R² = {glu3["r_squared_adj"]:.4f}, Reduced Chi² = {glu3["chi2_red"]:.1e}")
plt.gca().invert_xaxis()
plt.show()



# Calculate the area under each peak
peak_areas = []
for i in range(7):
    peak_area = (
        np.trapezoid(glu3["amplitudes"][i] * voigt_profile(sorted(x) - glu3["centers"][i], glu3["sigmas"][i], glu3["gammas"][i]), sorted(x))
        / area* 100
    )
    peak_areas.append(peak_area)
    print(f"Area under Peak {glu3["centers"][i]:.2f}: {peak_area}")

#taking the data for visualisation
for i in range(7):
    glu3[f"area_{i}"] = glu3["amplitudes"][i] * voigt_profile(sorted(x) - glu3["centers"][i], glu3["sigmas"][i], glu3["gammas"][i])
glu3["wavenumber"] = xy["wavenumber"]
glu3["spectra"] = xy["baseline_corrected"] 