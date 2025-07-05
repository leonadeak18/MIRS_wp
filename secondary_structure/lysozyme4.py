# Deconvolution of FTIR data
import ftir_secondary_structure as ss
import pandas as pd
import numpy as np
from scipy.special import voigt_profile
import matplotlib.pyplot as plt
import importlib
importlib.reload(ss)

# Load the FTIR data
data = pd.read_csv("data/lysozyme4.csv", header=0)
data.rename(columns={data.columns[0]: "wavenumber"}, inplace=True)

# Preprocess the data
pre_data = ss.preprocessing(data, window=11, normalize=False)
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
    0.00001, 1685, 5, 5, # Peak 1
    0.0001, 1675, 5, 5, # Peak 3
    0.0002, 1656, 2, 2, # Peak 4
    0.00003, 1647, 1, 1, # Peak 5
    0.00001, 1642, 1, 1, # Peak 6
    0.00001, 1633, 1, 1, # Peak 7
    0.0001, 1615, 1, 1, # Peak 8
]

lower_bounds = [
    #Amplitude, Center, Sigma, Gamma
    0, initial_params[1]- 2, 0, 0,  # Peak1 
    0, initial_params[5] - 2, 0, 0,  # Peak3
    0, initial_params[9] - 2, 0, 0,  # Peak4
    0, initial_params[13] - 2, 0, 0,  # Peak5
    0, initial_params[17] - 1, 0, 0,  # Peak6
    0, initial_params[21] - 2, 0, 0,  # Peak7
    0, initial_params[25] - 2, 0, 0,  # Peak8
]
upper_bounds = [
    #Amplitude, Center, Sigma, Gamma
    np.inf, initial_params[1] + 2, 10, 10,  # Peak1
    np.inf, initial_params[5] + 2, 7, 7,  # Peak3
    np.inf, initial_params[9] + 2, 10, 10,  # Peak4
    np.inf, initial_params[13] + 0, 10, 10,  # Peak5
    np.inf, initial_params[17] + 1, 10, 10,  # Peak6
    np.inf, initial_params[21] + 2, 10, 10,  # Peak7
    np.inf, initial_params[25] + 2, 10, 10,  # Peak8

]
lys4 = ss.deconvolution(x, y, initial_params, lower_bounds, upper_bounds)

# Plot Results
plt.figure(figsize=(7, 5))
plt.plot(x, y, "bo", markersize=3, label="Experimental Data")
plt.plot(x, lys4["y_fit"], "r-", linewidth=2, label="Total Fit")
colors = ["g", "m", "c", "y", "k", "orange", "brown", "purple", "pink", "gray"]
for i in range(len(lys4["centers"])):
    plt.plot(
        x,
        lys4["amplitudes"][i] * voigt_profile(x - lys4["centers"][i], lys4["sigmas"][i], lys4["gammas"][i]),
        colors[i],
        linestyle="--",
        label=f"Peak {i + 1}",
    )
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorbance (a.u.)")
plt.legend()
plt.title(f"adjusted R² = {lys4["r_squared_adj"]:.4f}, Reduced Chi² = {lys4["chi2_red"]:.1e}")
plt.gca().invert_xaxis()
plt.show()



# Calculate the area under each peak
peak_areas = []
for i in range(len(lys4["centers"])):
    peak_area = (
        np.trapezoid(lys4["amplitudes"][i] * voigt_profile(sorted(x) - lys4["centers"][i], lys4["sigmas"][i], lys4["gammas"][i]), sorted(x))
        / area* 100
    )
    peak_areas.append(peak_area)
    print(f"Area under Peak {lys4["centers"][i]:.2f}: {peak_area}")
    
    
#taking the data for visualisation
for i in range(5):
    lys4[f"area_{i}"] = lys4["amplitudes"][i] * voigt_profile(sorted(x) - lys4["centers"][i], lys4["sigmas"][i], lys4["gammas"][i])
    