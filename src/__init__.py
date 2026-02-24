"""
Extended Kalman Filter for BME Sensor Data
==========================================

A comprehensive EKF implementation for filtering BME280/BME680 sensor data.

This package provides:
- ExtendedKalmanFilter: Generic EKF implementation
- AdaptiveEKF: Self-tuning EKF with automatic noise estimation
- BMESensorEKF: EKF optimized for BME280 (T, H, P)
- BME680GasEKF: Extended EKF for BME680 (T, H, P, Gas)

Why EKF instead of Standard Kalman Filter?
-----------------------------------------
Standard Kalman Filter requires:
1. Linear state transition: x(k+1) = A*x(k) + B*u(k)
2. Time-based prediction: typically uses Δt for state evolution
3. Linear measurements: z(k) = H*x(k)

BME sensors violate these assumptions:
1. Non-linear sensor responses (temperature-humidity coupling)
2. No meaningful time-based state evolution model
3. Non-linear calibration curves

The EKF handles these by:
1. Using non-linear functions f(x) and h(x)
2. Linearizing via Jacobian matrices at each step
3. Modeling sensor physics instead of time evolution

Usage:
------
    from extended_kalman_filter import BMESensorEKF
    
    # Create filter
    ekf = BMESensorEKF(
        initial_temperature=25.0,
        initial_humidity=50.0,
        initial_pressure=1013.25
    )
    
    # Process readings
    while True:
        raw_T, raw_H, raw_P = read_bme_sensor()
        filtered = ekf.update(raw_T, raw_H, raw_P)
        print(f"Filtered T: {filtered['temperature']:.2f}°C")
"""

# Handle both relative and absolute imports
try:
    from .ekf_core import (
        ExtendedKalmanFilter,
        AdaptiveEKF,
        compute_numerical_jacobian
    )
    from .bme_ekf import (
        BMESensorEKF,
        BME680GasEKF
    )
except ImportError:
    from ekf_core import (
        ExtendedKalmanFilter,
        AdaptiveEKF,
        compute_numerical_jacobian
    )
    from bme_ekf import (
        BMESensorEKF,
        BME680GasEKF
    )

__all__ = [
    'ExtendedKalmanFilter',
    'AdaptiveEKF',
    'BMESensorEKF',
    'BME680GasEKF',
    'compute_numerical_jacobian'
]

__version__ = '1.0.0'
__author__ = 'BME Sensor Fusion System'
