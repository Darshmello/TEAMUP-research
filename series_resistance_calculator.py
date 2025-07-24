# series_resistance_calculator.py
import numpy as np
from scipy.stats import linregress
from data_loader import load_pv_data

# Physical constants (from outside sources, generally accepted values)
Q_CHARGE = 1.60218e-19  # Elementary charge in Coulombs [C]
K_BOLTZMANN = 1.38065e-23 # Boltzmann constant in Joules/Kelvin [J/K]

def calculate_sites_rs(v_curve, i_curve, isc_val, i_peak, v_peak, temperature_k):
    """
    Calculates series resistance (Rs) using the Sites method as described in Sites notes.pdf [1].
    Plots dV/dI on the y-axis vs. 1/(I + Isc) on the x-axis, and Rs is the y-intercept.

    Args:
        v_curve (np.array): Array of voltage points (V).
        i_curve (np.array): Array of current points (A).
        isc_val (float): Short-circuit current (Isc) from summary data, used as Iph (photocurrent).
        i_peak (float): Current at maximum power point (Ipeak).
        v_peak (float): Voltage at maximum power point (Vpeak).
        temperature_k (float): Temperature in Kelvin for calculating Vo (nkT/q).

    Returns:
        float: The calculated series resistance (Rs) in Ohms.
        float: The ideality factor 'n' derived from the fit's slope.
        tuple: (x_values, y_values) used for the linear fit.
        str: A message regarding the calculation outcome or issues.
    """
    # 1. Filter data for the operating region (positive voltage and current)
    # The Sites method for high-current region benefits from data near Isc.
    # We filter out data points with non-positive current or voltage,
    # as they are not relevant for the positive part of the IV curve for power generation
    # or for ln(I+Iph) calculation.
    positive_indices = np.where((v_curve > 0) & (i_curve > 0))
    v_segment = v_curve[positive_indices]
    i_segment = i_curve[positive_indices]

    if len(v_segment) < 2:
        return np.nan, np.nan, ([], []), "Not enough valid data points in the positive quadrant."

    # --- Filtering commented out below ---
    # Sort data by current to ensure monotonicity for differentiation
    # (If V_curve is monotonically increasing, I_curve will not be through MPP)
    #sort_indices = np.argsort(i_segment)
    #i_segment = i_segment[sort_indices]
    #v_segment = v_segment[sort_indices]

    # Further filter for the linear region (high current) based on Sites' suggestion
    # "By making J large, 1/J becomes small, the data fall closer to the origin" [1]
    # This implies selecting points where current (I) is high, typically towards Isc.
    # Let's take points where current is above 50% of Isc, up to the maximum current observed in this segment.
    # This range attempts to capture the 'linear' region where Rs dominates, avoiding noise at very low currents.
    #if isc_val > 0:
    #    current_threshold_low = isc_val * 0.5 # Example threshold, can be adjusted
    #else: # If isc is zero or negative (e.g. from bad measurement), set a small positive threshold
    #    current_threshold_low = 1e-6 # A very small positive current

    # Find the range to include high currents, excluding potentially noisy very low currents.
    # We'll take all positive current points if the highest current is below a small value like 0.005A,
    # otherwise, we will limit the range from 0.0005A to Ipeak.
    # This is an heuristic choice based on observing sample data range.
    #if np.max(i_segment) < 0.005: # For very low current devices
    #    segment_indices = np.where(i_segment > current_threshold_low)
    #else: # For devices with higher currents
    #    segment_indices = np.where(i_segment > 0.0005) # Starting from a reasonable positive current

    # Use all positive current points (no further filtering)
    segment_indices = np.arange(len(i_segment))

    if len(segment_indices) < 2:
        return np.nan, np.nan, ([], []), "Not enough data points in the selected high-current region for differentiation."

    i_fit_segment = i_segment[segment_indices]
    v_fit_segment = v_segment[segment_indices]

    if len(i_fit_segment) < 2:
        return np.nan, np.nan, ([], []), "Not enough data points after high-current filtering."

    # 2. Calculate dV/dI (or dV/dJ in Sites' notation) and 1/(I + Iph)
    # Numerical differentiation (finite difference approximation)
    delta_v = np.diff(v_fit_segment)
    delta_i = np.diff(i_fit_segment)

    # Filter out points where delta_i is zero or very close to zero to avoid division by zero
    non_zero_delta_i_indices = np.where(np.abs(delta_i) > 1e-10) # Adjust tolerance as needed
    if len(non_zero_delta_i_indices[0]) == 0:
        return np.nan, np.nan, ([], []), "All current differences are zero, cannot perform dV/dI."

    dv_di = delta_v[non_zero_delta_i_indices] / delta_i[non_zero_delta_i_indices]
    
    # Calculate average current for the x-axis points
    i_avg_for_x = (i_fit_segment[:-1] + i_fit_segment[1:])[non_zero_delta_i_indices]

    # Calculate x-axis values: 1 / (I_avg + Iph). Use isc_val as Iph.
    # Ensure denominator is not zero or too small
    x_values = 1 / (i_avg_for_x + isc_val)
    
    # 3. Perform linear regression on x_values and dv_di
    if len(x_values) < 2:
        return np.nan, np.nan, ([], []), "Not enough points to perform linear regression after differentiation and filtering."

    try:
        slope, intercept, r_value, p_value, std_err = linregress(x_values, dv_di)
    except ValueError as ve:
        return np.nan, np.nan, ([], []), f"Linear regression error (e.g., all same x values): {ve}"

    # Rs is the y-intercept of the plot [1]
    rs = intercept

    # Vo = nkT/q is the slope of the plot [1]
    # We can derive the ideality factor 'n' if temperature is known
    # Temperature_AI is assumed to be in Celsius, convert to Kelvin
    try:
        temp_kelvin = temperature_k # Assuming already Kelvin, or correct if in C.
        if temp_kelvin < 100: # Simple heuristic to check if likely Celsius
             temp_kelvin = temp_kelvin + 273.15 # Convert to Kelvin if likely Celsius
        else:
             temp_kelvin = temperature_k # Assume it's already Kelvin if large value
    except TypeError:
        # Handle cases where temperature_k might be None or non-numeric
        temp_kelvin = 298.15 # Default to 25 deg C (298.15 K) if no valid temp given
        print(f"Warning: Invalid temperature value provided, defaulting to {temp_kelvin}K for ideality factor calculation.")

    if temp_kelvin == 0 or slope == 0: # Avoid division by zero
        ideality_factor = np.nan
        message = "Rs calculated. Ideality factor 'n' could not be calculated (zero temperature or slope)."
    else:
        ideality_factor = (slope * Q_CHARGE) / (K_BOLTZMANN * temp_kelvin)
        message = "Rs and ideality factor 'n' calculated successfully."

    return rs, ideality_factor, (x_values, dv_di), message


if __name__ == "__main__":
    # IMPORTANT: Replace 'Ch102_2025_05_18wCalculations.xlsx' with the actual path to your file
    file_name = 'Ch102_2025_05_18wCalculations.xlsx'
    pv_data_records = load_pv_data(file_name)

    if not pv_data_records:
        print("No data loaded. Exiting.")
    else:
        print("\n--- Calculating Series Resistance for each measurement ---")
        for i, measurement in enumerate(pv_data_records):
            print(f"\nMeasurement {i+1} (Date/Time: {measurement['date_time']}, Name: {measurement['name']})")
            
            # Print number of data points in 'volts_curve' and 'amps_curve' to check data loading
            print(f"  Number of voltage points: {len(measurement['volts_curve'])}")
            print(f"  Number of current points: {len(measurement['amps_curve'])}")
            # Extract required parameters for calculation
            v_curve_arr = measurement['volts_curve']
            i_curve_arr = measurement['amps_curve']
            isc = measurement['isc']
            v_peak = measurement['v_peak']
            i_peak = measurement['i_peak']
            
            # Using Temperature_AI for cell temperature. Assuming it's in Celsius and converting to Kelvin.
            # If Temperature_AI is missing or invalid, default to 25C (298.15K)
            temp_ai = measurement.get('temperature_ai', 25.0)
            if not isinstance(temp_ai, (int, float)):
                print("Warning: 'Temperature_AI' not a valid number, defaulting to 25.0 Celsius.")
                temp_ai = 25.0
            
            # Pass temperature in Celsius, let the function convert to Kelvin internally
            # The Sites notes [1] mention temperature, and Pyscha et al. [6] mention 25C correction.
            # For the constant Vo=nkT/q, T must be in Kelvin.
            
            calculated_rs, ideality_n, (x_fit, y_fit), status_message = calculate_sites_rs(
                v_curve_arr, i_curve_arr, isc, i_peak, v_peak, temp_ai
            )

            print(f"Status: {status_message}")
            if not np.isnan(calculated_rs):
                print(f"  Calculated Series Resistance (Rs): {calculated_rs:.6f} Ohms")
                if not np.isnan(ideality_n):
                    print(f"  Derived Ideality Factor (n): {ideality_n:.3f}")
            else:
                print("  Series Resistance could not be calculated.")
                
            # Optional: You can plot x_fit and y_fit here to visually inspect the linear fit.
            import matplotlib.pyplot as plt
            if len(x_fit) > 1 and len(y_fit) > 1:
                plt.figure()
                plt.plot(x_fit, y_fit, 'o', label='Data Points for Fit')
                plt.plot(
                    x_fit,
                    calculated_rs + (ideality_n * K_BOLTZMANN * (temp_ai + 273.15)) / Q_CHARGE * x_fit,
                    label=f'Linear Fit: Rs={calculated_rs:.4f}'
                )
                plt.xlabel('1 / (I + Isc) [1/A]')
                plt.ylabel('dV/dI [Ohm]')
                plt.title(f"Sites Method for {measurement['name']}")
                plt.legend()
                plt.grid(True)
                plt.show()

