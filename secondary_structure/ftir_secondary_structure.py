##This function can help to deconvolve of FTIR peaks. This technique can be utilized to determine the secondary structure of proteins which exist on amide I (1700 - 1600 cm-1)
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, argrelmin
from scipy.special import voigt_profile
from scipy.optimize import curve_fit


def preprocessing(input, window = 7, normalize=False):
    """This function works by
    1. removing data outside the range of 1600 to 1701 cm-1
    2. applying a second derivative with  points window Savitzky-Golay filter
    3. creating an inverted second derivative
    4. Normalizing the inverted second derivative
    """

    # applying second derivative with two-degree polynomial order and 7-points Savitzky-Golay filter
    input["sec_der"] = savgol_filter(input.iloc[:,1], window_length= window, polyorder=2, deriv=2)
    
    # removing data outside of amide I
    input = input[(input["wavenumber"] >= 1600) & (input["wavenumber"] <= 1701)].copy()

    # creating inverted second derivative
    input["inv_sec_der"] = -1 * input["sec_der"]
    
    #normalization
    if normalize:
        min_val = input["inv_sec_der"].min()
        max_val = input["inv_sec_der"].max()
        input["inv_sec_der"] = (input["inv_sec_der"] - min_val) / (max_val - min_val)
        
    return input[["wavenumber", "inv_sec_der"]]

def valley_peaks(input):
    """
    This function works by finding the local minima in the inverted second derivative
    """
    ### Find local minima using on inverted second derivative
    minima_indices = argrelmin(input["inv_sec_der"].values, order=5)[0]
    minima = input.iloc[minima_indices]
    
    ### edge baseline anchors
    left_edge = input[(input["wavenumber"] >= 1600) & (input["wavenumber"] <= 1610)].nsmallest(1, "inv_sec_der")
    right_edge = input[(input["wavenumber"] >= 1690) & (input["wavenumber"] <= 1700)].nsmallest(1, "inv_sec_der")
    
        
    # Filter middle minima that are lower than edge minima values
    middle_minima = minima[(minima["wavenumber"] > 1610) & (minima["wavenumber"] < 1690)]
    middle_minima = middle_minima.nsmallest(4, "inv_sec_der")
    
    # Combine baseline anchor points
    baseline_points = pd.concat([left_edge, middle_minima, right_edge]).sort_values("wavenumber")
    return baseline_points

def baseline_correction(input, baseline_points):
    """This function works by interpolating the baseline points and subtracting it from the inverted second derivative"""  
    # Interpolate baseline
    baseline = np.interp(input["wavenumber"], baseline_points["wavenumber"], baseline_points["inv_sec_der"])

    # Baseline correction
    input["baseline_corrected"] = input["inv_sec_der"] - baseline
    return input[["wavenumber", "baseline_corrected"]]    
    


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
