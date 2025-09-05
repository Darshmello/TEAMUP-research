"""
Complete IV Curve Analysis with Sites Method
============================================
All-in-one script for analyzing IV curves and generating all graphs.
Just run: python iv_analysis_complete.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    """Configuration parameters"""
    ELEMENTARY_CHARGE = 1.602e-19  # Coulombs
    BOLTZMANN_CONSTANT = 1.381e-23  # J/K
    TEMPERATURE_KELVIN = 298  # Room temperature
    Z_SCORE_THRESHOLD = 3
    SMOOTHING_WINDOW = 5
    MIN_FORWARD_POINTS = 5
    MPP_FRACTION = 1.1
    FIGURE_DPI = 150
    DEFAULT_AREA_CM2 = 0.1
    OUTPUT_FOLDER = 'analysis_results'
    
    # Column patterns for auto-detection
    VOLTAGE_PATTERNS = ['V', 'Voltage', 'Bias', 'V(V)', 'V (V)', 'Potential']
    CURRENT_PATTERNS = ['I', 'Current', 'I(A)', 'I (A)', 'J', 'Current Density']

config = Config()

# Create output folder
if not os.path.exists(config.OUTPUT_FOLDER):
    os.makedirs(config.OUTPUT_FOLDER)

# ==============================================================================
# DATA EXTRACTION AND CLEANING
# ==============================================================================
class DataProcessor:
    """Handles data extraction and cleaning"""
    
    def __init__(self, filename):
        self.filename = filename
        self.raw_data = {}
        self.clean_data = {}
        self.params = {}
        
    def extract_data(self, sheet_name=None, voltage_col=None, current_col=None, area_cm2=None):
        """Extract IV data from Excel file"""
        print(f"\nExtracting data from: {self.filename}")
        
        # Read Excel file
        if sheet_name is None:
            df = pd.read_excel(self.filename)
        else:
            df = pd.read_excel(self.filename, sheet_name=sheet_name)
        
        # Auto-detect columns if not specified
        if voltage_col is None or current_col is None:
            for col in df.columns:
                col_str = str(col).strip()
                
                if voltage_col is None:
                    for pattern in config.VOLTAGE_PATTERNS:
                        if pattern.lower() in col_str.lower():
                            voltage_col = col
                            print(f"  Auto-detected voltage column: '{col}'")
                            break
                
                if current_col is None:
                    for pattern in config.CURRENT_PATTERNS:
                        if pattern.lower() in col_str.lower():
                            current_col = col
                            print(f"  Auto-detected current column: '{col}'")
                            break
        
        # Extract data
        voltage = df[voltage_col].dropna().values
        current = df[current_col].dropna().values
        
        # Ensure same length
        min_len = min(len(voltage), len(current))
        voltage = voltage[:min_len]
        current = current[:min_len]
        
        # Convert to current density if area provided
        if area_cm2:
            current_density = current / area_cm2 * 1000  # mA/cm²
        else:
            current_density = current
        
        # Sort by voltage
        sort_idx = np.argsort(voltage)
        self.raw_data = {
            'voltage': voltage[sort_idx],
            'current': current[sort_idx],
            'current_density': current_density[sort_idx]
        }
        
        print(f"  Extracted {len(voltage)} data points")
        print(f"  V range: {voltage.min():.3f} to {voltage.max():.3f} V")
        print(f"  J range: {current_density.min():.1f} to {current_density.max():.1f} mA/cm²")
        
        return self.raw_data
    
    def remove_outliers(self, z_threshold=3):
        """Remove outliers using z-score method"""
        V = self.raw_data['voltage']
        J = self.raw_data['current_density']
        
        # Z-score outlier detection
        z_scores_v = np.abs(stats.zscore(V))
        z_scores_j = np.abs(stats.zscore(J))
        
        # Keep points that are not outliers
        mask = (z_scores_v < z_threshold) & (z_scores_j < z_threshold)
        
        self.clean_data = {
            'voltage': V[mask],
            'current_density': J[mask]
        }
        
        removed = len(V) - len(self.clean_data['voltage'])
        print(f"  Removed {removed} outliers ({removed/len(V)*100:.1f}%)")
        
        return self.clean_data
    
    def calculate_parameters(self):
        """Calculate key IV parameters"""
        data = self.clean_data if self.clean_data else self.raw_data
        V = data['voltage']
        J = data['current_density']
        
        # Find key points
        oc_idx = np.argmin(np.abs(J))
        V_oc = V[oc_idx]
        
        sc_idx = np.argmin(np.abs(V))
        J_sc = J[sc_idx]
        J_ph = -J_sc  # Photocurrent
        
        # Maximum power point
        power = V * J
        mpp_idx = np.argmax(np.abs(power))
        V_mpp = V[mpp_idx]
        J_mpp = J[mpp_idx]
        P_max = np.abs(V_mpp * J_mpp)
        
        # Fill factor
        FF = P_max / (np.abs(V_oc * J_sc)) if V_oc != 0 and J_sc != 0 else 0
        
        self.params = {
            'V_oc': V_oc,
            'J_sc': J_sc,
            'J_ph': J_ph,
            'V_mpp': V_mpp,
            'J_mpp': J_mpp,
            'P_max': P_max,
            'FF': FF
        }
        
        return self.params

# ==============================================================================
# SITES METHOD ANALYSIS
# ==============================================================================
class SitesAnalysis:
    """Performs Sites method analysis"""
    
    def __init__(self, data_processor):
        self.processor = data_processor
        self.dV_dJ = None
        self.inverse_current = None
        self.sites_results = None
        
    def calculate_differential_resistance(self):
        """Calculate dV/dJ"""
        data = self.processor.clean_data if self.processor.clean_data else self.processor.raw_data
        params = self.processor.params
        
        V = data['voltage']
        J = data['current_density']
        
        # Use forward bias region (beyond MPP)
        # First try points with positive current
        forward_mask = J > 0
        
        # If not enough points, use points beyond Voc
        if np.sum(forward_mask) < config.MIN_FORWARD_POINTS:
            forward_mask = V > params['V_oc'] * 0.8
        
        # If still not enough, use points beyond MPP
        if np.sum(forward_mask) < config.MIN_FORWARD_POINTS:
            forward_mask = V > params['V_mpp']
        
        V_forward = V[forward_mask]
        J_forward = J[forward_mask]
        
        # Check if we have enough points
        if len(V_forward) < 3:
            print(f"  Warning: Only {len(V_forward)} forward bias points available")
            # Use all positive voltage points as fallback
            forward_mask = V > 0
            V_forward = V[forward_mask]
            J_forward = J[forward_mask]
        
        if len(V_forward) < 2:
            raise ValueError("Not enough data points for Sites analysis. Need more forward bias measurements.")
        
        print(f"  Using {len(V_forward)} points for Sites analysis")
        
        # Calculate derivative
        dV_dJ = np.gradient(V_forward, J_forward)
        
        # Handle any inf or nan values
        valid_mask = np.isfinite(dV_dJ)
        dV_dJ = dV_dJ[valid_mask]
        J_forward = J_forward[valid_mask]
        V_forward = V_forward[valid_mask]
        
        # Smooth if needed
        if len(dV_dJ) > config.SMOOTHING_WINDOW:
            dV_dJ = savgol_filter(dV_dJ, min(len(dV_dJ), config.SMOOTHING_WINDOW), min(2, len(dV_dJ)-1))
        
        self.dV_dJ = dV_dJ
        self.J_forward = J_forward
        self.V_forward = V_forward
        
        return dV_dJ
    
    def calculate_inverse_current(self):
        """Calculate 1/(J + Jph)"""
        J_ph = self.processor.params['J_ph']
        
        # Calculate 1/(J + Jph)
        epsilon = 1e-10
        denominator = self.J_forward + J_ph + epsilon
        
        # Filter valid points
        valid_mask = (denominator > epsilon) & np.isfinite(self.dV_dJ)
        
        if np.sum(valid_mask) < 2:
            raise ValueError("Not enough valid points for Sites method fitting")
        
        self.inverse_current = 1.0 / denominator[valid_mask]
        self.dV_dJ_valid = self.dV_dJ[valid_mask]
        self.J_forward_valid = self.J_forward[valid_mask]
        
        return self.inverse_current
    
    def perform_sites_fit(self):
        """Perform linear regression for Sites method"""
        if self.inverse_current is None:
            self.calculate_differential_resistance()
            self.calculate_inverse_current()
        
        x = self.inverse_current
        y = self.dV_dJ_valid
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Extract parameters
        R_series = intercept  # Series resistance
        n_calculated = slope * config.ELEMENTARY_CHARGE / (config.BOLTZMANN_CONSTANT * config.TEMPERATURE_KELVIN)
        
        self.sites_results = {
            'R_series': R_series,
            'n': n_calculated,
            'r_squared': r_value**2,
            'slope': slope,
            'intercept': intercept,
            'x_data': x,
            'y_data': y
        }
        
        return self.sites_results

# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_all_results(processor, sites_analysis):
    """Generate all plots"""
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Get data
    data = processor.clean_data if processor.clean_data else processor.raw_data
    params = processor.params
    sites_results = sites_analysis.sites_results
    
    # 1. IV Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(processor.raw_data['voltage'], processor.raw_data['current_density'], 
            'o', alpha=0.3, color='gray', markersize=3, label='Raw data')
    ax1.plot(data['voltage'], data['current_density'], 
            'b-', linewidth=2, label='Clean data')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.plot(params['V_oc'], 0, 'ro', markersize=8, label=f"Voc={params['V_oc']:.3f}V")
    ax1.plot(0, params['J_sc'], 'go', markersize=8, label=f"Jsc={params['J_sc']:.1f}mA/cm²")
    ax1.set_xlabel('Voltage (V)', fontsize=12)
    ax1.set_ylabel('Current Density (mA/cm²)', fontsize=12)
    ax1.set_title('IV Characteristic Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 2. Power Curve
    ax2 = plt.subplot(2, 3, 2)
    power = data['voltage'] * np.abs(data['current_density'])
    ax2.plot(data['voltage'], power, 'r-', linewidth=2)
    ax2.plot(params['V_mpp'], params['P_max'], 'ko', markersize=10, 
            label=f"MPP: {params['P_max']:.2f} mW/cm²")
    ax2.set_xlabel('Voltage (V)', fontsize=12)
    ax2.set_ylabel('Power Density (mW/cm²)', fontsize=12)
    ax2.set_title(f'Power Curve (FF={params["FF"]:.3f})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. dV/dJ vs J
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(sites_analysis.J_forward, sites_analysis.dV_dJ, 'g-', linewidth=2)
    ax3.axvline(x=params['J_mpp'], color='r', linestyle='--', alpha=0.5, 
               label=f"J_mpp={params['J_mpp']:.1f}")
    ax3.set_xlabel('Current Density (mA/cm²)', fontsize=12)
    ax3.set_ylabel('dV/dJ (Ω·cm²)', fontsize=12)
    ax3.set_title('Differential Resistance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 1/(J+Jph) vs J
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(sites_analysis.J_forward_valid, 
            sites_analysis.inverse_current, 'mo-', linewidth=2, markersize=4)
    ax4.set_xlabel('Current Density (mA/cm²)', fontsize=12)
    ax4.set_ylabel('1/(J + Jph) (cm²/mA)', fontsize=12)
    ax4.set_title(f'Inverse Current (Jph={params["J_ph"]:.1f} mA/cm²)', 
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. MAIN SITES PLOT: dV/dJ vs 1/(J+Jph)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(sites_results['x_data'], sites_results['y_data'], 
            'bo', markersize=8, alpha=0.7, label='Data')
    
    # Fit line
    x_fit = np.linspace(np.min(sites_results['x_data']), 
                       np.max(sites_results['x_data']), 100)
    y_fit = sites_results['slope'] * x_fit + sites_results['intercept']
    ax5.plot(x_fit, y_fit, 'r-', linewidth=2.5, 
            label=f'Fit (R²={sites_results["r_squared"]:.4f})')
    
    # Mark intercept (Rs)
    ax5.axhline(y=sites_results['R_series'], color='green', linestyle='--', 
               alpha=0.5, linewidth=1.5)
    ax5.text(0.05, 0.95, 
            f'Rs = {sites_results["R_series"]:.4f} Ω·cm²\nn = {sites_results["n"]:.2f}',
            transform=ax5.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax5.set_xlabel('1/(J + Jph) (cm²/mA)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('dV/dJ (Ω·cm²)', fontsize=12, fontweight='bold')
    ax5.set_title('SITES METHOD ANALYSIS', fontsize=14, fontweight='bold', color='darkred')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""
    ANALYSIS RESULTS
    {'='*30}
    
    Device Parameters:
    • Voc = {params['V_oc']:.4f} V
    • Jsc = {params['J_sc']:.3f} mA/cm²
    • FF = {params['FF']:.3f}
    • Pmax = {params['P_max']:.2f} mW/cm²
    
    Sites Method Results:
    • Rs = {sites_results['R_series']:.4f} Ω·cm²
    • n = {sites_results['n']:.2f}
    • R² = {sites_results['r_squared']:.4f}
    
    Data Quality:
    • Points used: {len(sites_results['x_data'])}
    • Outliers removed: {len(processor.raw_data['voltage']) - len(data['voltage'])}
    """
    
    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('IV Curve Analysis - Sites Method for Series Resistance Extraction', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(config.OUTPUT_FOLDER, 'sites_analysis_complete.png')
    plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    plt.show()
    
    return fig

# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================
def analyze_iv_curve(filename, area_cm2=0.1, sheet_name=None):
    """
    Main function to analyze IV curve and generate all plots.
    
    Parameters:
    -----------
    filename : str
        Path to Excel file with IV data
    area_cm2 : float
        Device area in cm²
    sheet_name : str, optional
        Specific sheet to analyze
    
    Returns:
    --------
    dict : Analysis results
    """
    
    print("\n" + "="*60)
    print("IV CURVE ANALYSIS - SITES METHOD")
    print("="*60)
    
    # Step 1: Extract and clean data
    processor = DataProcessor(filename)
    processor.extract_data(sheet_name=sheet_name, area_cm2=area_cm2)
    processor.remove_outliers()
    
    # Step 2: Calculate parameters
    params = processor.calculate_parameters()
    print("\nDevice Parameters:")
    print(f"  Voc = {params['V_oc']:.4f} V")
    print(f"  Jsc = {params['J_sc']:.3f} mA/cm²")
    print(f"  FF = {params['FF']:.3f}")
    print(f"  Pmax = {params['P_max']:.2f} mW/cm²")
    
    # Step 3: Perform Sites analysis
    sites = SitesAnalysis(processor)
    sites_results = sites.perform_sites_fit()
    
    print("\nSites Method Results:")
    print(f"  Series Resistance Rs = {sites_results['R_series']:.4f} Ω·cm²")
    print(f"  Ideality Factor n = {sites_results['n']:.2f}")
    print(f"  R² = {sites_results['r_squared']:.4f}")
    
    # Step 4: Generate all plots
    plot_all_results(processor, sites)
    
    # Step 5: Save results to CSV
    results_df = pd.DataFrame([{
        'filename': filename,
        'Voc_V': params['V_oc'],
        'Jsc_mA_cm2': params['J_sc'],
        'FF': params['FF'],
        'Pmax_mW_cm2': params['P_max'],
        'Rs_ohm_cm2': sites_results['R_series'],
        'n_ideality': sites_results['n'],
        'R_squared': sites_results['r_squared']
    }])
    
    csv_file = os.path.join(config.OUTPUT_FOLDER, 'analysis_results.csv')
    results_df.to_csv(csv_file, index=False)
    print(f"\n✓ Results saved to: {csv_file}")
    
    return {
        'params': params,
        'sites_results': sites_results,
        'processor': processor,
        'sites_analysis': sites
    }

# ==============================================================================
# CREATE SAMPLE DATA (for testing)
# ==============================================================================
def create_sample_data():
    """Create realistic sample IV curve data for testing"""
    print("\nCreating sample IV curve data...")
    
    # Generate voltage range
    V = np.linspace(-0.2, 0.8, 150)
    
    # Realistic parameters for a solar cell
    n = 1.5  # Ideality factor
    kT = config.BOLTZMANN_CONSTANT * config.TEMPERATURE_KELVIN
    q = config.ELEMENTARY_CHARGE
    kT_q = kT / q  # Thermal voltage (~0.026V at room temp)
    
    # Realistic device parameters
    Jo = 1e-12  # Saturation current density in A/cm²
    Jph = 0.035  # Photocurrent in A/cm² (35 mA/cm²)
    Rs = 2.0  # Series resistance in Ohm·cm²
    Rsh = 1000  # Shunt resistance in Ohm·cm²
    
    # Generate current using realistic diode equation with series and shunt resistance
    I = np.zeros_like(V)
    
    for i, v in enumerate(V):
        # Newton-Raphson iteration to solve implicit equation
        # I = Jph - Jo*(exp((V+I*Rs)/(n*kT/q)) - 1) - (V+I*Rs)/Rsh
        
        i_guess = 0
        for iteration in range(50):
            v_diode = v + i_guess * Rs
            
            # Current from diode equation
            if v_diode / (n * kT_q) < 60:  # Prevent overflow
                i_diode = Jo * (np.exp(v_diode / (n * kT_q)) - 1)
            else:
                i_diode = Jo * np.exp(60)
            
            # Total current
            i_new = Jph - i_diode - v_diode / Rsh
            
            # Check convergence
            if abs(i_new - i_guess) < 1e-10:
                break
            
            # Update guess with damping for stability
            i_guess = 0.7 * i_guess + 0.3 * i_new
        
        I[i] = i_guess
    
    # Convert to mA/cm²
    J = I * 1000
    
    # Add small realistic noise
    J += np.random.normal(0, 0.05, len(J))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Voltage (V)': V,
        'Current Density (mA/cm²)': J
    })
    
    # Save to Excel
    if not os.path.exists('data'):
        os.makedirs('data')
    
    filename = 'data/sample_iv_curve.xlsx'
    df.to_excel(filename, index=False)
    print(f"✓ Sample data created: {filename}")
    print(f"  Expected Rs ≈ {Rs:.1f} Ω·cm²")
    print(f"  Expected n ≈ {n:.1f}")
    
    return filename

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================
if __name__ == "__main__":
    import sys
    
    # Check if file is provided as argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        area = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    else:
        # Create and use sample data
        print("\nNo file specified. Creating sample data for demonstration...")
        filename = create_sample_data()
        area = 1.0  # Sample data is already in current density
    
    # Run analysis
    try:
        results = analyze_iv_curve(filename, area_cm2=area)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Check the '{config.OUTPUT_FOLDER}' folder for:")
        print("  • sites_analysis_complete.png - All plots")
        print("  • analysis_results.csv - Numerical results")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()