##This function can help to deconvolve of FTIR peaks. This technique can be utilized to determine the secondary structure of proteins which exist on amide I (1700 - 1600 cm-1)
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.special import voigt_profile
from scipy.optimize import curve_fit


def preprocessing(input):
    """This function works by
    1. removing data outside the range of 1600 to 1701 cm-1
    2. applying a second derivative with a 7-point window Savitzky-Golay filter
    3. creating an inverted second derivative
    4. baseline correction"""

    # applying second derivative with two-degree polynomial order and 7-points Savitzky-Golay filter
    input["sec_der"] = savgol_filter(input["Absorbance"], 7, 2, deriv=2)
    
    #smoothing the second derivative
    #input["sec_der"] = savgol_filter(input["sec_der"], 9, 2)
    
    # removing data outside of amide I
    input = input[(input["wavenumber"] >= 1600) & (input["wavenumber"] <= 1701)]

    # creating inverted second derivative
    input["inv_sec_der"] = -1 * input["sec_der"]

    # baseline correction
    ### identify the lowest points near 1700 and 1600 cm-1
    low_point_1700 = input.loc[
        (input["wavenumber"] >= 1690) & (input["wavenumber"] <= 1700)
    ].nsmallest(1, "inv_sec_der")
    low_point_1600 = input.loc[
        (input["wavenumber"] >= 1600) & (input["wavenumber"] <= 1610)
    ].nsmallest(1, "inv_sec_der")

    # Extract wavenumbers and intensities
    baseline_x = [low_point_1600["wavenumber"].values[0], low_point_1700["wavenumber"].values[0]]
    baseline_y = [low_point_1600["inv_sec_der"].values[0], low_point_1700["inv_sec_der"].values[0]]

    # Interpolate the baseline
    baseline = np.interp(input["wavenumber"], baseline_x, baseline_y)

    # Subtract the interpolated baseline
    input["baseline_corrected"] = input["inv_sec_der"] - baseline
    output = pd.concat([input["wavenumber"], input["baseline_corrected"]], axis=1)

    return output


def total_area(input, w1, w2, col_x=0, col_y=1):
    """this function works by calculating the area under the curve between two wavenumbers
    Parameters:
    - input: DataFrame containing the spectral data
    - w1, w2: Wavenumber range for integration
    - col_x: Index of the wavenumber column (default: 0)
    - col_y: Index of the absorbance/intensity column (default: 1)
    """
    filtered_data = input[(input.iloc[:, col_x] >= w1) & (input.iloc[:, col_y] <= w2)]
    filtered_data = filtered_data.sort_values(by=input.columns[col_x])
    area = np.trapz(filtered_data.iloc[:, col_y], filtered_data.iloc[:, col_x])

    return area


def deconvolution(x, y, initial_params, lower_bounds, upper_bounds):
    """this function works by fitting a Voigt profile to the filtered data using scipy.optimize.curve_fit
    Parameters:
    - x: Wavenumber data
    - y: Absorbance data
    - initial_params: Initial parameters for the Voigt profile
    - lower_bounds: Lower bounds for the Voigt profile parameters
    - upper_bounds: Upper bounds for the Voigt profile parameters
    """

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

    # Perform the fitting using scipy's curve_fit
    popt, pcov = curve_fit(
        multi_voigt, x, y, p0=initial_params, bounds=(lower_bounds, upper_bounds)
    )

    # Extract fitted parameters
    amplitudes = popt[::4]
    centers = popt[1::4]
    sigmas = popt[2::4]
    gammas = popt[3::4]

    # Compute the fitted y-values
    y_fit = multi_voigt(x, *popt)

    # Compute R²
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compute Adjusted R²
    n = len(y)  # Number of observations
    p = len(popt)  # Number of parameters
    r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

    # Compute Reduced Chi-Square
    chi2 = np.sum((residuals**2) / (y_fit + 1e-6))  # Avoid division by zero
    dof = len(x) - len(popt)  # Degrees of freedom
    chi2_red = chi2 / dof

    return {
        "amplitudes": amplitudes,
        "centers": centers,
        "sigmas": sigmas,
        "gammas": gammas,
        "y_fit": y_fit,
        "r_squared": r_squared,
        "r_squared_adj": r_squared_adj,
        "chi2_red": chi2_red,
    }
    
def deconvolution2(x, y, initial_params, lower_bounds, upper_bounds, max_attempts=1000, target_adj_r2=0.99):
    """Repeatedly fits Voigt profile to x and y until adjusted R² > target or max_attempts reached."""

    def multi_voigt(x, *params):
        n = len(params) // 4
        result = np.zeros_like(x)
        for i in range(n):
            amp = params[i * 4]
            center = params[i * 4 + 1]
            sigma = params[i * 4 + 2]
            gamma = params[i * 4 + 3]
            result += amp * voigt_profile(x - center, sigma, gamma)
        return result

    best_fit = None
    best_adj_r2 = -np.inf
    attempt = 0

    while best_adj_r2 < target_adj_r2 and attempt < max_attempts:
        try:
            # Slightly perturb initial parameters randomly within bounds
            perturbed_init = np.clip(
                initial_params + np.random.normal(0, 0.1, size=len(initial_params)),
                lower_bounds, upper_bounds
            )

            popt, _ = curve_fit(
                multi_voigt, x, y, p0=perturbed_init, bounds=(lower_bounds, upper_bounds)
            )

            y_fit = multi_voigt(x, *popt)
            residuals = y - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            n_obs = len(y)
            n_params = len(popt)
            r_squared_adj = 1 - ((1 - r_squared) * (n_obs - 1) / (n_obs - n_params - 1))

            if r_squared_adj > best_adj_r2:
                best_adj_r2 = r_squared_adj
                best_fit = {
                    "amplitudes": popt[::4],
                    "centers": popt[1::4],
                    "sigmas": popt[2::4],
                    "gammas": popt[3::4],
                    "y_fit": y_fit,
                    "r_squared": r_squared,
                    "r_squared_adj": r_squared_adj,
                    "chi2_red": np.sum((residuals**2) / (y_fit + 1e-6)) / (n_obs - n_params)
                }
        except Exception as e:
            pass  # Skip failed fits

        attempt += 1

    if best_fit is None:
        raise RuntimeError("Failed to achieve a valid fit after all attempts.")

    return best_fit
