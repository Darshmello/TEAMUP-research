# data_loader.py
import pandas as pd
import numpy as np
import ast # Used for safely evaluating string representations of lists

def load_pv_data(file_path, sheet_name='Sheet1'):
    """
    Loads photovoltaic (PV) data from an XLSX file, parses IV curve strings
    into numerical arrays, and organizes all relevant data.

    Args:
        file_path (str): The path to the XLSX file.
        sheet_name (str, optional): The name of the sheet to load. Defaults to 'Sheet1'.

    Returns:
        list of dict: A list where each dictionary represents a measurement
                      record containing arrays for IV curves and other key parameters.
                      Returns an empty list if the file cannot be read or no data.
    """
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

    # List to hold processed data for each measurement
    processed_measurements = []

    # Iterate through each row (measurement) in the DataFrame
    for index, row in df.iterrows():
        try:
            # Parse 'volts_curve' and 'amps_curve' which are expected to be string representations of lists
            # Using ast.literal_eval for safe evaluation of string literals
            volts_curve = np.array(ast.literal_eval(row['volts_curve']))
            amps_curve = np.array(ast.literal_eval(row['amps_curve']))

            # Store relevant summary data and the parsed IV curves
            measurement_data = {
                'date_time': row['Date_Time'],
                'name': row['Name'],
                'peak_power': row['PeakPower'],
                'v_peak': row['Vpeak'],
                'i_peak': row['Ipeak'],
                'isc': row['Isc'], # Short-circuit current, used as Jph in Sites method
                'voc': row['Voc'], # Open-circuit voltage
                'temperature_ai': row['Temperature_AI'], # Assuming this is the relevant temperature for the cell
                'volts_curve': volts_curve,
                'amps_curve': amps_curve
            }
            processed_measurements.append(measurement_data)
        except KeyError as ke:
            print(f"Skipping row {index} due to missing column: {ke}. Ensure 'volts_curve', 'amps_curve', 'Isc', 'Vpeak', etc., exist.")
            continue
        except ValueError as ve:
            print(f"Skipping row {index} due to parsing error in IV curve data: {ve}. Data might not be a valid list string.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred processing row {index}: {e}")
            continue

    print(f"Successfully loaded data for {len(processed_measurements)} measurements.")
    return processed_measurements

if __name__ == "__main__":
    # Example usage:
    # IMPORTANT: Replace 'Ch102_2025_05_18wCalculations.xlsx' with the actual path to your file
    file_name = 'Ch102_2025_05_18wCalculations.xlsx'
    pv_data = load_pv_data(file_name)

    if pv_data:
        # Print some info about the first loaded measurement
        print("\n--- First Measurement Data Sample ---")
        first_measurement = pv_data
        print(f"Date/Time: {first_measurement['date_time']}")
        print(f"Name: {first_measurement['name']}")
        print(f"Vpeak: {first_measurement['v_peak']:.4f}, Ipeak: {first_measurement['i_peak']:.4f}")
        print(f"Isc: {first_measurement['isc']:.4f}, Voc: {first_measurement['voc']:.4f}")
        print(f"First 5 V points: {first_measurement['volts_curve'][:5]}")
        print(f"First 5 I points: {first_measurement['amps_curve'][:5]}")