# Deconvolution of FTIR data
import ftir_secondary_structure as ss
import pandas as pd
import numpy as np
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

# Load the FTIR data
data = pd.read_csv("data/alb1.csv", header=0)
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
    0.0001, 1680, 13, 0.001, # Peak 1
    0.00020, 1656, 11.5, 7, # Peak 2
    0.0001, 1638, 7, 0.1, # Peak 3
    0.0001, 1627, 7, 0.1, # Peak 4
    0.00007, 1614, 7, 0.001, # Peak 5
]

lower_bounds = [
    #Amplitude, Center, Sigma, Gamma
    0, initial_params[1]- 2, 0, 0,  # Peak1 
    0, initial_params[5] - 2, 0, 0,  # Peak2
    0, initial_params[9] - 2, 0, 0,  # Peak3
    0, initial_params[13] - 2, 0, 0,  # Peak4
    0, initial_params[17] - 2, 0, 0,  # Peak5
]
upper_bounds = [
    #Amplitude, Center, Sigma, Gamma
    np.inf, initial_params[1] + 2, 50, 50,  # Peak1
    np.inf, initial_params[5] + 2, 50, 50,  # Peak2
    np.inf, initial_params[9] + 2, 50, 50,  # Peak3
    np.inf, initial_params[13] + 2, 50, 50,  # Peak4
    np.inf, initial_params[17] + 2, 50, 50,  # Peak5
]
alb1 = ss.deconvolution(x, y, initial_params, lower_bounds, upper_bounds)

# Plot Results
plt.figure(figsize=(7, 5))
plt.plot(x, y, "bo", markersize=3, label="Experimental Data")
plt.plot(x, alb1["y_fit"], "r-", linewidth=2, label="Total Fit")
colors = ["g", "m", "c", "y", "k"]
for i in range(5):
    plt.plot(
        x,
        alb1["amplitudes"][i] * voigt_profile(x - alb1["centers"][i], alb1["sigmas"][i], alb1["gammas"][i]),
        colors[i],
        linestyle="--",
        label=f"Peak {i + 1}",
    )
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorbance (a.u.)")
plt.legend()
plt.title(f"adjusted R² = {alb1["r_squared_adj"]:.4f}, Reduced Chi² = {alb1["chi2_red"]:.1e}")
plt.gca().invert_xaxis()
plt.show()



# Calculate the area under each peak
peak_areas = []
for i in range(5):
    peak_area = (
        np.trapezoid(alb1["amplitudes"][i] * voigt_profile(sorted(x) - alb1["centers"][i], alb1["sigmas"][i], alb1["gammas"][i]), sorted(x))
        / area* 100
    )
    peak_areas.append(peak_area)
    print(f"Area under Peak {alb1["centers"][i]:.2f}: {peak_area}")
    
    
#taking the data for visualisation
for i in range(5):
    alb1[f"area_{i}"] = alb1["amplitudes"][i] * voigt_profile(sorted(x) - alb1["centers"][i], alb1["sigmas"][i], alb1["gammas"][i])
alb1["wavenumber"] = xy["wavenumber"]
alb1["spectra"] = xy["baseline_corrected"]