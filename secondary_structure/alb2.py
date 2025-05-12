# Deconvolution of FTIR data
import ftir_secondary_structure as ss
import pandas as pd
import numpy as np
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

# Load the FTIR data
data = pd.read_csv("data/alb2.csv", header=0)
data.rename(columns={data.columns[0]: "wavenumber"}, inplace=True)

# Preprocess the data
processed_data = ss.preprocessing(data)

# Calculate the total area under the curve between 1600 and 1700
wavenumber_zero = processed_data[processed_data["baseline_corrected"] == 0]
area = ss.total_area(processed_data, w2 = wavenumber_zero.iloc[0,0], w1=wavenumber_zero.iloc[1,0])

# deconvoluted the FTIR peaks
xy = processed_data[(processed_data["wavenumber"] <= wavenumber_zero.iloc[0,0]) & (processed_data["wavenumber"] >= wavenumber_zero.iloc[1,0])]
x = xy["wavenumber"]
y = xy["baseline_corrected"]

## Provide initial parameter guesses for 4 peaks
initial_params = [
    #Amplitude, Center, Sigma, Gamma
    0.0001, 1680, 13, 0.001, # Peak 1
    0.00020, 1656, 11.5, 7, # Peak 2
    0.00003, 1631, 18, 3.7, # Peak 3
    0.00007, 1614, 7, 0.001, # Peak 4
]
lower_bounds = [
    #Amplitude, Center, Sigma, Gamma
    0, initial_params[1] - 2, 0, 0,  # Peak1 
    0, initial_params[5] - 2, 0, 0,  # Peak2
    0, initial_params[9] - 2, 0, 0,  # Peak3
    0, initial_params[13] - 2, 0, 0,  # Peak4
]
upper_bounds = [
    #Amplitude, Center, Sigma, Gamma
    np.inf, initial_params[1] + 2, 50, 50,  # Peak1
    np.inf, initial_params[5] + 2, 50, 50,  # Peak2
    np.inf, initial_params[9] + 2, 50, 50,  # Peak3
    np.inf, initial_params[13] + 2, 50, 50,  # Peak4
]
alb2 = ss.deconvolution(x, y, initial_params, lower_bounds, upper_bounds)

# Plot Results
plt.figure(figsize=(7, 5))
plt.plot(x, y, "bo", markersize=3, label="Experimental Data")
plt.plot(x, alb2["y_fit"], "r-", linewidth=2, label="Total Fit")
colors = ["g", "m", "c", "y"]
for i in range(4):
    plt.plot(
        x,
        alb2["amplitudes"][i] * voigt_profile(x - alb2["centers"][i], alb2["sigmas"][i], alb2["gammas"][i]),
        colors[i],
        linestyle="--",
        label=f"Peak {i + 1}",
    )
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorbance (a.u.)")
plt.legend()
plt.title(f"adjusted R² = {alb2["r_squared_adj"]:.4f}, Reduced Chi² = {alb2["chi2_red"]:.1e}")
plt.gca().invert_xaxis()
plt.show()



# Calculate the area under each peak
peak_areas = []
for i in range(4):
    peak_area = (
        np.trapz(alb2["amplitudes"][i] * voigt_profile(sorted(x) - alb2["centers"][i], alb2["sigmas"][i], alb2["gammas"][i]), sorted(x))
        / area* 100
    )
    peak_areas.append(peak_area)
    print(f"Area under Peak {alb2["centers"][i]:.2f}: {peak_area}")