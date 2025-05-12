import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

# Load the FTIR data
data = pd.read_csv("alb2.csv", header=0)
data.rename(columns={data.columns[0]: "wavenumber"}, inplace=True)

# filtered data in the range amide I (1700 - 1600 cm-1)
data = data[(data["wavenumber"] >= 1600) & (data["wavenumber"] <= 1701)]

# apply second derivative with a 7-point window Savitzky-Golay filter
data["sec_der"] = savgol_filter(data["Absorbance"], 7, 2, deriv=2)

# create inverted second derivative
data["inv_sec_der"] = -1 * data["sec_der"]

# Identify the lowest points near 1700 and 1600
low_point_1700 = data.loc[
    (data["wavenumber"] >= 1695) & (data["wavenumber"] <= 1700)
].nsmallest(1, "inv_sec_der")
low_point_1600 = data.loc[
    (data["wavenumber"] >= 1600) & (data["wavenumber"] <= 1605)
].nsmallest(1, "inv_sec_der")

# Combine the points for linear regression
baseline_points = pd.concat([low_point_1700, low_point_1600])

# Perform linear regression
X = baseline_points["wavenumber"].values.reshape(-1, 1)
y = baseline_points["inv_sec_der"].values
model = LinearRegression()
model.fit(X, y)
baseline = model.predict(data["wavenumber"].values.reshape(-1, 1))

# Subtract the linear baseline from the inverted second derivative
data["baseline_corrected"] = data["inv_sec_der"] - baseline

# Calculate the area under the curve between 1695 and 1605
# Sort the filtered data by wavenumber in ascending order
filtered_data = data[(data["wavenumber"] >= 1605) & (data["wavenumber"] <= 1695)]
filtered_data = filtered_data.sort_values(by="wavenumber")
area = np.trapz(filtered_data["baseline_corrected"], filtered_data["wavenumber"])
print(f"Area under the curve between 1695 and 1605: {area}")

# Plot the filtered data to visualize the area under the curve
plt.plot(
    filtered_data["wavenumber"],
    filtered_data["baseline_corrected"],
    label="Filtered Baseline Corrected",
)
plt.fill_between(
    filtered_data["wavenumber"], filtered_data["baseline_corrected"], alpha=0.3
)
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Baseline Corrected Absorbance")
plt.title("Filtered Baseline Corrected FTIR Spectrum")
plt.legend()
plt.show()


# Fit a Voigt profile to the filtered data using scipy.optimize.curve_fit
# Extract data
x = filtered_data["wavenumber"].values
y = filtered_data["baseline_corrected"].values


# Define the Voigt profile function (sum of multiple peaks)
def multi_voigt(x, *params):
    n = len(params) // 4
    result = np.zeros_like(x)
    for i in range(n):
        amplitude = params[i * 4]
        center = params[i * 4 + 1]
        sigma = params[i * 4 + 2]
        gamma = params[i * 4 + 3]
        result += amplitude * voigt_profile(x - center, sigma, gamma)
    return result


# Provide initial parameter guesses for 4 peaks
initial_params = [
    0.0001,
    1680,
    13,
    0.001,
    0.00020,
    1656,
    11.5,
    7,
    0.00003,
    1631,
    18,
    3.7,
    0.00007,
    1614,
    7,
    0.001,
]

# define the bounds for the parameters
lower_bounds = [
    0,
    1680 - 2,
    0,
    0,  # amp1, cen1, sigma1, gamma1
    0,
    1656 - 2,
    0,
    0,  # amp2, cen2, sigma2, gamma2
    0,
    1631 - 2,
    0,
    0,  # amp3, cen3, sigma3, gamma3
    0,
    1614 - 2,
    0,
    0,  # amp4, cen4, sigma4, gamma4
]
upper_bounds = [
    np.inf,
    1680 + 2,
    50,
    50,  # amp1, cen1, sigma1, gamma1
    np.inf,
    1656 + 2,
    50,
    50,  # amp2, cen2, sigma2, gamma2
    np.inf,
    1631 + 2,
    50,
    50,  # amp3, cen3, sigma3, gamma3
    np.inf,
    1614 + 2,
    50,
    50,  # amp4, cen4, sigma4, gamma4
]

# Perform the fitting using scipy's curve_fit
popt, pcov = curve_fit(
    multi_voigt, x, y, p0=initial_params, bounds=(lower_bounds, upper_bounds)
)

# Extract fitted parameters
amplitudes = popt[::4]
centers = popt[1::4]
sigmas = popt[2::4]
gammas = popt[3::4]

# Print fitted parameters
print("Fitted Parameters:")
for i in range(len(amplitudes)):
    print(
        f"Peak {i + 1}: Amplitude = {amplitudes[i]}, Center = {centers[i]}, Sigma = {sigmas[i]}, Gamma = {gammas[i]}"
    )

# Compute the fitted y-values
y_fit = multi_voigt(x, *popt)

# Ensure y_fit values are non-negative
# y_fit = np.maximum(y_fit, 0)

# Compute R²
residuals = y - y_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Compute Adjusted R²
n = len(y)  # Number of observations
p = len(popt)  # Number of parameters
r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
print(r_squared_adj)

# Compute Reduced Chi-Square
chi2 = np.sum((residuals**2) / (y_fit + 1e-6))  # Avoid division by zero
dof = len(x) - len(popt)  # Degrees of freedom
chi2_red = chi2 / dof

# Plot Results
plt.figure(figsize=(7, 5))
plt.plot(x, y, "bo", markersize=3, label="Experimental Data")
plt.plot(x, y_fit, "r-", linewidth=2, label="Total Fit")
colors = ["g", "m", "c", "y"]
for i in range(4):
    plt.plot(
        x,
        amplitudes[i] * voigt_profile(x - centers[i], sigmas[i], gammas[i]),
        colors[i],
        linestyle="--",
        label=f"Peak {i + 1}",
    )
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Normalized Absorbance")
plt.legend()
plt.title(f"adjusted R² = {r_squared_adj:.4f}, Reduced Chi² = {chi2_red:.1e}")
plt.gca().invert_xaxis()  # Standard FTIR visualization
plt.show()


# Calculate the area under each peak
peak_areas = []
for i in range(4):
    peak_area = (
        np.trapz(amplitudes[i] * voigt_profile(x - centers[i], sigmas[i], gammas[i]), x)
        / area
        * 100
    )
    peak_areas.append(peak_area)
    print(f"Area under Peak {i + 1}: {peak_area}")
