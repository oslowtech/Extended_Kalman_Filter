"""
Extended Kalman Filter Usage Examples
=====================================

This module demonstrates how to use the EKF for BME sensor filtering.
Examples include:
1. Basic BME280 filtering
2. BME680 with gas resistance
3. Simulated noisy data filtering
4. Real-time monitoring
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekf_core import ExtendedKalmanFilter, AdaptiveEKF, compute_numerical_jacobian
from src.bme_ekf import BMESensorEKF, BME680GasEKF


def example_1_basic_filtering():
    """
    Example 1: Basic BME280 Filtering
    
    Demonstrates filtering noisy temperature, humidity, and pressure readings.
    """
    print("=" * 60)
    print("Example 1: Basic BME280 Filtering")
    print("=" * 60)
    
    # Create filter with initial estimates
    ekf = BMESensorEKF(
        initial_temperature=25.0,
        initial_humidity=50.0,
        initial_pressure=1013.25,
        temperature_noise=0.5,    # Sensor accuracy ±0.5°C
        humidity_noise=3.0,       # Sensor accuracy ±3%
        pressure_noise=1.0        # Sensor accuracy ±1 hPa
    )
    
    # Simulate sensor readings with noise
    np.random.seed(42)
    true_temp = 22.5
    true_humidity = 65.0
    true_pressure = 1015.0
    
    print(f"\nTrue values: T={true_temp}°C, H={true_humidity}%, P={true_pressure}hPa")
    print("\nFiltering 20 noisy readings:")
    print("-" * 60)
    
    for i in range(20):
        # Simulate noisy measurements
        noisy_T = true_temp + np.random.normal(0, 0.5)
        noisy_H = true_humidity + np.random.normal(0, 3.0)
        noisy_P = true_pressure + np.random.normal(0, 1.0)
        
        # Filter the readings
        result = ekf.update(noisy_T, noisy_H, noisy_P)
        
        if i % 4 == 0:  # Print every 4th reading
            print(f"Step {i+1:2d}: Raw T={noisy_T:6.2f}°C -> Filtered T={result['temperature']:6.2f}°C "
                  f"(±{result['temperature_std']:.2f})")
    
    # Final estimates
    T, H, P = ekf.get_filtered_values()
    T_bias, H_bias, P_bias = ekf.get_sensor_biases()
    
    print("-" * 60)
    print(f"\nFinal filtered values:")
    print(f"  Temperature: {T:.2f}°C (true: {true_temp}°C, error: {abs(T-true_temp):.2f}°C)")
    print(f"  Humidity:    {H:.2f}%  (true: {true_humidity}%, error: {abs(H-true_humidity):.2f}%)")
    print(f"  Pressure:    {P:.2f}hPa (true: {true_pressure}hPa, error: {abs(P-true_pressure):.2f}hPa)")
    print(f"\nEstimated sensor biases:")
    print(f"  T_bias: {T_bias:+.3f}°C, H_bias: {H_bias:+.3f}%, P_bias: {P_bias:+.3f}hPa")


def example_2_biased_sensor():
    """
    Example 2: Filtering with Sensor Bias
    
    Demonstrates EKF's ability to estimate and compensate for sensor bias.
    """
    print("\n" + "=" * 60)
    print("Example 2: Sensor Bias Estimation")
    print("=" * 60)
    
    # Create filter
    ekf = BMESensorEKF()
    
    # Simulate sensor with known bias
    true_temp = 25.0
    sensor_bias = 1.5  # Sensor reads 1.5°C too high
    
    print(f"\nTrue temperature: {true_temp}°C")
    print(f"Sensor bias: +{sensor_bias}°C (sensor reads high)")
    print("\nFiltering 50 readings to estimate bias:")
    
    for i in range(50):
        # Biased + noisy measurements
        biased_T = true_temp + sensor_bias + np.random.normal(0, 0.3)
        stable_H = 50.0 + np.random.normal(0, 1.0)
        stable_P = 1013.25 + np.random.normal(0, 0.5)
        
        result = ekf.update(biased_T, stable_H, stable_P)
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1:2d}: Estimated T_bias = {result['temperature_bias']:+.3f}°C")
    
    T, _, _ = ekf.get_filtered_values()
    T_bias_est, _, _ = ekf.get_sensor_biases()
    
    print(f"\nFinal Results:")
    print(f"  Estimated temperature: {T:.2f}°C (should be ~{true_temp + sensor_bias - T_bias_est:.1f}°C)")
    print(f"  Estimated bias: {T_bias_est:+.3f}°C (actual: +{sensor_bias}°C)")
    print(f"  Bias estimation error: {abs(T_bias_est - sensor_bias):.3f}°C")


def example_3_temperature_change():
    """
    Example 3: Tracking Temperature Changes
    
    Demonstrates EKF's ability to track gradual environmental changes.
    """
    print("\n" + "=" * 60)
    print("Example 3: Tracking Temperature Changes")
    print("=" * 60)
    
    # Create filter
    ekf = BMESensorEKF(initial_temperature=20.0)
    
    print("\nSimulating gradual temperature rise from 20°C to 30°C:")
    
    temperatures = []
    filtered_temps = []
    
    for i in range(100):
        # True temperature rises gradually
        true_temp = 20.0 + (10.0 * i / 99)  # Linear rise
        
        # Noisy measurement
        noisy_T = true_temp + np.random.normal(0, 0.5)
        noisy_H = 50.0 + np.random.normal(0, 2.0)
        noisy_P = 1013.25 + np.random.normal(0, 0.5)
        
        result = ekf.update(noisy_T, noisy_H, noisy_P)
        
        temperatures.append(true_temp)
        filtered_temps.append(result['temperature'])
        
        if (i + 1) % 20 == 0:
            print(f"  Step {i+1:3d}: True T={true_temp:.1f}°C, "
                  f"Measured T={noisy_T:.2f}°C, "
                  f"Filtered T={result['temperature']:.2f}°C")
    
    # Calculate tracking error
    tracking_error = np.mean([abs(f - t) for f, t in zip(filtered_temps, temperatures)])
    print(f"\nAverage tracking error: {tracking_error:.3f}°C")


def example_4_bme680_with_gas():
    """
    Example 4: BME680 with Gas Resistance and IAQ
    
    Demonstrates filtering gas sensor data for indoor air quality.
    """
    print("\n" + "=" * 60)
    print("Example 4: BME680 Gas Sensor Filtering")
    print("=" * 60)
    
    # Create BME680 filter
    ekf = BME680GasEKF(
        initial_temperature=25.0,
        initial_humidity=50.0,
        initial_pressure=1013.25,
        initial_gas_resistance=50000.0,
        gas_noise=5000.0
    )
    
    print("\nSimulating gas sensor data with air quality changes:")
    
    # Baseline period (good air)
    print("\n--- Phase 1: Good Air Quality ---")
    for i in range(20):
        result = ekf.update_with_gas(
            temperature=25.0 + np.random.normal(0, 0.3),
            humidity=50.0 + np.random.normal(0, 2.0),
            pressure=1013.25 + np.random.normal(0, 0.5),
            gas_resistance=50000.0 + np.random.normal(0, 3000)
        )
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1:2d}: Gas R={result['gas_resistance']:.0f}Ω, IAQ={result['iaq']:.0f}")
    
    # VOC event (resistance drops)
    print("\n--- Phase 2: VOC Event (Poor Air) ---")
    for i in range(20, 40):
        result = ekf.update_with_gas(
            temperature=25.0 + np.random.normal(0, 0.3),
            humidity=50.0 + np.random.normal(0, 2.0),
            pressure=1013.25 + np.random.normal(0, 0.5),
            gas_resistance=30000.0 + np.random.normal(0, 3000)  # Lower resistance = VOCs
        )
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1:2d}: Gas R={result['gas_resistance']:.0f}Ω, IAQ={result['iaq']:.0f}")
    
    # Recovery period
    print("\n--- Phase 3: Air Quality Recovery ---")
    for i in range(40, 60):
        result = ekf.update_with_gas(
            temperature=25.0 + np.random.normal(0, 0.3),
            humidity=50.0 + np.random.normal(0, 2.0),
            pressure=1013.25 + np.random.normal(0, 0.5),
            gas_resistance=45000.0 + np.random.normal(0, 3000)
        )
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1:2d}: Gas R={result['gas_resistance']:.0f}Ω, IAQ={result['iaq']:.0f}")


def example_5_diagnostics():
    """
    Example 5: Filter Diagnostics
    
    Demonstrates how to monitor filter performance.
    """
    print("\n" + "=" * 60)
    print("Example 5: Filter Diagnostics")
    print("=" * 60)
    
    ekf = BMESensorEKF()
    
    # Run filter for a few samples
    for i in range(10):
        ekf.update(
            25.0 + np.random.normal(0, 0.5),
            50.0 + np.random.normal(0, 2.0),
            1013.25 + np.random.normal(0, 0.5)
        )
    
    # Get diagnostics
    diag = ekf.get_diagnostics()
    
    print("\nFilter Diagnostics:")
    print(f"  Innovation (residual): T={diag['innovation'][0]:.3f}°C, "
          f"H={diag['innovation'][1]:.3f}%, P={diag['innovation'][2]:.3f}hPa")
    print(f"  Normalized Innovation Squared (NIS): {diag['normalized_innovation_squared']:.3f}")
    print(f"  (NIS should be ~3.0 for 3 measurements, chi-squared distributed)")
    
    print(f"\n  Ambient Estimates:")
    print(f"    Temperature: {diag['ambient_estimates']['temperature']:.2f}°C")
    print(f"    Humidity: {diag['ambient_estimates']['humidity']:.2f}%")
    print(f"    Pressure: {diag['ambient_estimates']['pressure']:.2f}hPa")
    
    # State uncertainties
    T_std, H_std, P_std = ekf.get_uncertainties()
    print(f"\n  State Uncertainties (1-sigma):")
    print(f"    Temperature: ±{T_std:.3f}°C")
    print(f"    Humidity: ±{H_std:.3f}%")
    print(f"    Pressure: ±{P_std:.3f}hPa")


def example_6_comparison_raw_vs_filtered():
    """
    Example 6: Raw vs Filtered Comparison
    
    Demonstrates noise reduction quantitatively.
    """
    print("\n" + "=" * 60)
    print("Example 6: Noise Reduction Analysis")
    print("=" * 60)
    
    ekf = BMESensorEKF()
    
    true_temp = 23.0
    noise_std = 1.0  # Noisy sensor
    
    raw_readings = []
    filtered_readings = []
    
    # Collect 100 samples
    for i in range(100):
        noisy_T = true_temp + np.random.normal(0, noise_std)
        result = ekf.update(noisy_T, 50.0, 1013.25)
        
        raw_readings.append(noisy_T)
        filtered_readings.append(result['temperature'])
    
    # Skip first 10 samples (filter warm-up)
    raw_readings = raw_readings[10:]
    filtered_readings = filtered_readings[10:]
    
    # Calculate statistics
    raw_std = np.std(raw_readings)
    filtered_std = np.std(filtered_readings)
    raw_error = np.mean([abs(r - true_temp) for r in raw_readings])
    filtered_error = np.mean([abs(f - true_temp) for f in filtered_readings])
    
    print(f"\nTrue temperature: {true_temp}°C")
    print(f"Measurement noise: ±{noise_std}°C (1-sigma)")
    print(f"\nResults after 100 samples:")
    print(f"  Raw readings:")
    print(f"    Standard deviation: {raw_std:.3f}°C")
    print(f"    Mean absolute error: {raw_error:.3f}°C")
    print(f"  Filtered readings:")
    print(f"    Standard deviation: {filtered_std:.3f}°C")
    print(f"    Mean absolute error: {filtered_error:.3f}°C")
    print(f"\n  Noise reduction: {(1 - filtered_std/raw_std) * 100:.1f}%")
    print(f"  Error reduction: {(1 - filtered_error/raw_error) * 100:.1f}%")


def example_7_micropython_usage():
    """
    Example 7: MicroPython Usage Pattern
    
    Shows how to use the filter in embedded systems.
    """
    print("\n" + "=" * 60)
    print("Example 7: MicroPython/Embedded Usage Pattern")
    print("=" * 60)
    
    print("""
# MicroPython example for ESP32 with BME280

from machine import I2C, Pin
from bme280 import BME280  # Your BME280 driver
from bme_ekf import BMESensorEKF

# Initialize hardware
i2c = I2C(0, scl=Pin(22), sda=Pin(21))
bme = BME280(i2c=i2c)

# Initialize EKF
ekf = BMESensorEKF(
    initial_temperature=25.0,
    initial_humidity=50.0,
    initial_pressure=1013.25
)

# Main loop
while True:
    # Read raw sensor values
    raw_T, raw_P, raw_H = bme.read_compensated_data()
    raw_T /= 100  # Convert to °C
    raw_H /= 1024  # Convert to %
    raw_P /= 256 / 100  # Convert to hPa
    
    # Filter the readings
    filtered = ekf.update(raw_T, raw_H, raw_P)
    
    # Use filtered values
    print("T: {:.1f}°C, H: {:.0f}%, P: {:.1f}hPa".format(
        filtered['temperature'],
        filtered['humidity'],
        filtered['pressure']
    ))
    
    time.sleep(1)
""")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("   EXTENDED KALMAN FILTER FOR BME SENSORS")
    print("   Usage Examples")
    print("=" * 60)
    
    example_1_basic_filtering()
    example_2_biased_sensor()
    example_3_temperature_change()
    example_4_bme680_with_gas()
    example_5_diagnostics()
    example_6_comparison_raw_vs_filtered()
    example_7_micropython_usage()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
