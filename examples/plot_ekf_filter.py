"""
EKF Filter Visualization
========================

Reads dummy sensor data from CSV, applies Extended Kalman Filter,
and plots raw vs filtered values to demonstrate noise reduction.

This script demonstrates how the EKF:
1. Smooths noisy measurements
2. Tracks changing environmental conditions
3. Reduces measurement variance while following true values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bme_ekf import BMESensorEKF


def load_sensor_data(csv_path: str) -> pd.DataFrame:
    """Load dummy sensor data from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def apply_ekf_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply EKF filter to raw sensor readings.
    
    Returns DataFrame with additional columns for filtered values.
    """
    # Initialize filter with first reading
    ekf = BMESensorEKF(
        initial_temperature=df['temperature_raw'].iloc[0],
        initial_humidity=df['humidity_raw'].iloc[0],
        initial_pressure=df['pressure_raw'].iloc[0],
        temperature_noise=0.5,
        humidity_noise=2.0,
        pressure_noise=0.5,
        process_noise_factor=0.5,  # More responsive to changes
        adaptive=False
    )
    
    # Storage for filtered values
    filtered_T = []
    filtered_H = []
    filtered_P = []
    uncertainty_T = []
    uncertainty_H = []
    uncertainty_P = []
    
    # Process each reading
    for idx, row in df.iterrows():
        result = ekf.update(
            row['temperature_raw'],
            row['humidity_raw'],
            row['pressure_raw']
        )
        
        filtered_T.append(result['temperature'])
        filtered_H.append(result['humidity'])
        filtered_P.append(result['pressure'])
        uncertainty_T.append(result['temperature_std'])
        uncertainty_H.append(result['humidity_std'])
        uncertainty_P.append(result['pressure_std'])
    
    # Add filtered columns to DataFrame
    df['temperature_filtered'] = filtered_T
    df['humidity_filtered'] = filtered_H
    df['pressure_filtered'] = filtered_P
    df['temperature_uncertainty'] = uncertainty_T
    df['humidity_uncertainty'] = uncertainty_H
    df['pressure_uncertainty'] = uncertainty_P
    
    return df


def plot_comparison(df: pd.DataFrame, output_path: str = None):
    """
    Create comparison plot: Raw vs Filtered vs True values.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Extended Kalman Filter: Raw vs Filtered BME Sensor Data', 
                 fontsize=14, fontweight='bold')
    
    samples = df['sample']
    
    # ============== Temperature Plot ==============
    ax1 = axes[0]
    
    # Raw measurements (noisy)
    ax1.scatter(samples, df['temperature_raw'], 
                color='red', alpha=0.4, s=20, label='Raw Measurements', zorder=2)
    
    # True values
    ax1.plot(samples, df['temperature_true'], 
             color='green', linewidth=2, linestyle='--', label='True Value', zorder=3)
    
    # Filtered values with uncertainty band
    ax1.plot(samples, df['temperature_filtered'], 
             color='blue', linewidth=2, label='EKF Filtered', zorder=4)
    ax1.fill_between(samples, 
                     df['temperature_filtered'] - 2*df['temperature_uncertainty'],
                     df['temperature_filtered'] + 2*df['temperature_uncertainty'],
                     color='blue', alpha=0.2, label='95% Confidence', zorder=1)
    
    ax1.set_ylabel('Temperature (°C)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Temperature: Noise Reduction & Tracking', fontsize=11)
    
    # ============== Humidity Plot ==============
    ax2 = axes[1]
    
    ax2.scatter(samples, df['humidity_raw'], 
                color='red', alpha=0.4, s=20, label='Raw Measurements', zorder=2)
    ax2.plot(samples, df['humidity_true'], 
             color='green', linewidth=2, linestyle='--', label='True Value', zorder=3)
    ax2.plot(samples, df['humidity_filtered'], 
             color='blue', linewidth=2, label='EKF Filtered', zorder=4)
    ax2.fill_between(samples, 
                     df['humidity_filtered'] - 2*df['humidity_uncertainty'],
                     df['humidity_filtered'] + 2*df['humidity_uncertainty'],
                     color='blue', alpha=0.2, label='95% Confidence', zorder=1)
    
    ax2.set_ylabel('Humidity (%)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Humidity: Noise Reduction & Tracking', fontsize=11)
    
    # ============== Pressure Plot ==============
    ax3 = axes[2]
    
    ax3.scatter(samples, df['pressure_raw'], 
                color='red', alpha=0.4, s=20, label='Raw Measurements', zorder=2)
    ax3.plot(samples, df['pressure_true'], 
             color='green', linewidth=2, linestyle='--', label='True Value', zorder=3)
    ax3.plot(samples, df['pressure_filtered'], 
             color='blue', linewidth=2, label='EKF Filtered', zorder=4)
    ax3.fill_between(samples, 
                     df['pressure_filtered'] - 2*df['pressure_uncertainty'],
                     df['pressure_filtered'] + 2*df['pressure_uncertainty'],
                     color='blue', alpha=0.2, label='95% Confidence', zorder=1)
    
    ax3.set_xlabel('Sample Number', fontsize=11)
    ax3.set_ylabel('Pressure (hPa)', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Pressure: Noise Reduction & Tracking', fontsize=11)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate error statistics for raw vs filtered values.
    """
    # Skip first 10 samples (filter warm-up period)
    df_stable = df.iloc[10:].copy()
    
    stats = {}
    
    for var in ['temperature', 'humidity', 'pressure']:
        raw_col = f'{var}_raw'
        filtered_col = f'{var}_filtered'
        true_col = f'{var}_true'
        
        # Raw errors
        raw_error = df_stable[raw_col] - df_stable[true_col]
        raw_mae = np.mean(np.abs(raw_error))
        raw_std = np.std(raw_error)
        raw_rmse = np.sqrt(np.mean(raw_error**2))
        
        # Filtered errors
        filtered_error = df_stable[filtered_col] - df_stable[true_col]
        filtered_mae = np.mean(np.abs(filtered_error))
        filtered_std = np.std(filtered_error)
        filtered_rmse = np.sqrt(np.mean(filtered_error**2))
        
        # Improvement
        mae_improvement = (1 - filtered_mae / raw_mae) * 100 if raw_mae > 0 else 0
        std_improvement = (1 - filtered_std / raw_std) * 100 if raw_std > 0 else 0
        
        stats[var] = {
            'raw_mae': raw_mae,
            'raw_std': raw_std,
            'raw_rmse': raw_rmse,
            'filtered_mae': filtered_mae,
            'filtered_std': filtered_std,
            'filtered_rmse': filtered_rmse,
            'mae_improvement': mae_improvement,
            'std_improvement': std_improvement
        }
    
    return stats


def print_statistics(stats: dict):
    """Print formatted statistics table."""
    print("\n" + "=" * 70)
    print(" EKF FILTER PERFORMANCE STATISTICS")
    print("=" * 70)
    print(f"{'Variable':<12} {'Metric':<10} {'Raw':<12} {'Filtered':<12} {'Improvement':<12}")
    print("-" * 70)
    
    units = {'temperature': '°C', 'humidity': '%', 'pressure': 'hPa'}
    
    for var, data in stats.items():
        unit = units[var]
        print(f"{var.capitalize():<12} {'MAE':<10} {data['raw_mae']:.3f} {unit:<5} "
              f"{data['filtered_mae']:.3f} {unit:<5} {data['mae_improvement']:+.1f}%")
        print(f"{'':12} {'STD':<10} {data['raw_std']:.3f} {unit:<5} "
              f"{data['filtered_std']:.3f} {unit:<5} {data['std_improvement']:+.1f}%")
        print(f"{'':12} {'RMSE':<10} {data['raw_rmse']:.3f} {unit:<5} "
              f"{data['filtered_rmse']:.3f} {unit:<5}")
        print("-" * 70)
    
    print("\nMAE = Mean Absolute Error | STD = Standard Deviation | RMSE = Root Mean Square Error")
    print("Positive improvement = filtered is better than raw")
    print("=" * 70)


def main():
    """Main function to run the visualization."""
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'dummy_sensor_data.csv')
    plot_path = os.path.join(script_dir, 'ekf_filter_comparison.png')
    
    print("\n" + "=" * 70)
    print(" EXTENDED KALMAN FILTER VISUALIZATION")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading sensor data...")
    df = load_sensor_data(csv_path)
    
    # Apply filter
    print("\n2. Applying Extended Kalman Filter...")
    df = apply_ekf_filter(df)
    print(f"   Processed {len(df)} samples")
    
    # Calculate statistics
    print("\n3. Calculating performance statistics...")
    stats = calculate_statistics(df)
    print_statistics(stats)
    
    # Create plot
    print("\n4. Generating comparison plot...")
    plot_comparison(df, plot_path)
    
    # Save filtered data
    output_csv = os.path.join(script_dir, 'filtered_sensor_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n5. Filtered data saved to: {output_csv}")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
