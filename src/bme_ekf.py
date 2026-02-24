"""
Extended Kalman Filter for BME Sensors (BME280/BME680)
======================================================

This module implements a specialized EKF for filtering and fusing
data from BME family environmental sensors.

BME Sensor Characteristics:
--------------------------
- BME280: Temperature, Humidity, Pressure
- BME680: Temperature, Humidity, Pressure, Gas Resistance (IAQ)

Why Standard Kalman Filter Fails for BME Sensors:
------------------------------------------------
1. NON-LINEAR SENSOR BEHAVIOR:
   - Temperature affects humidity readings (cross-sensitivity)
   - Pressure measurements depend on temperature compensation
   - Sensor response curves are non-linear
   
2. TIME REFERENCE PROBLEM:
   Standard KF: x(k+1) = x(k) + velocity * Δt
   
   This assumes state changes linearly with time.
   BME sensors don't work this way:
   - Environmental conditions change non-linearly
   - Sensor samples may arrive at irregular intervals
   - No "velocity" makes sense for temperature/humidity
   
3. SENSOR DRIFT AND HYSTERESIS:
   - Sensors exhibit path-dependent behavior
   - Drift cannot be modeled with simple linear prediction

EKF Solution - VALUE-BASED FILTERING:
------------------------------------
Instead of: "How does state evolve over time?"
We ask: "How do sensor readings relate to true values?"

State Transition: Models sensor physics (thermal inertia, equilibration)
Measurement Model: Models sensor non-linearities (calibration curves)

This approach:
- Works WITHOUT time intervals
- Handles non-linear sensor responses
- Adapts to sensor-specific characteristics

Author: BME Sensor Fusion System
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, List, Dict

# Try relative import first (when used as package), fall back to absolute
try:
    from .ekf_core import ExtendedKalmanFilter, AdaptiveEKF, compute_numerical_jacobian
except ImportError:
    from ekf_core import ExtendedKalmanFilter, AdaptiveEKF, compute_numerical_jacobian


class BMESensorEKF:
    """
    Extended Kalman Filter optimized for BME280/BME680 sensors.
    
    State Vector:
    ------------
    x = [T, H, P, T_bias, H_bias, P_bias]^T
    
    Where:
        T: True temperature (°C)
        H: True relative humidity (%)
        P: True pressure (hPa)
        T_bias: Temperature sensor bias/drift
        H_bias: Humidity sensor bias/drift
        P_bias: Pressure sensor bias/drift
    
    Measurement Vector:
    ------------------
    z = [T_measured, H_measured, P_measured]^T
    
    State Transition Model (NO TIME DEPENDENCY):
    -------------------------------------------
    The state transition models sensor physics:
    
    T(k+1) = T(k) + α_T * (T_ambient - T(k))
    H(k+1) = H(k) + α_H * (H_ambient - H(k)) * f(T)
    P(k+1) = P(k) + α_P * (P_ambient - P(k))
    bias(k+1) = β * bias(k)  [Random walk with decay]
    
    Where:
        α: Equilibration rate (sensor thermal mass effect)
        β: Bias persistence factor
        f(T): Temperature-humidity coupling function
    
    These parameters model SENSOR BEHAVIOR, not time evolution.
    The filter works sample-by-sample regardless of timing.
    
    Measurement Model (NON-LINEAR):
    ------------------------------
    z_T = T + T_bias + ε_T
    z_H = H + H_bias + κ*(T - T_ref) + ε_H  [Temperature compensation]
    z_P = P + P_bias + ε_P
    
    Where κ is the humidity-temperature cross-sensitivity coefficient.
    """
    
    def __init__(
        self,
        initial_temperature: float = 25.0,
        initial_humidity: float = 50.0,
        initial_pressure: float = 1013.25,
        temperature_noise: float = 0.5,
        humidity_noise: float = 2.0,
        pressure_noise: float = 0.5,
        process_noise_factor: float = 0.1,
        adaptive: bool = True
    ):
        """
        Initialize BME sensor EKF.
        
        Args:
            initial_temperature: Initial temperature estimate (°C)
            initial_humidity: Initial humidity estimate (%)
            initial_pressure: Initial pressure estimate (hPa)
            temperature_noise: Temperature measurement noise std (°C)
            humidity_noise: Humidity measurement noise std (%)
            pressure_noise: Pressure measurement noise std (hPa)
            process_noise_factor: Process noise scaling (higher = more adaptive)
            adaptive: Use adaptive noise estimation
        """
        self.n_states = 6  # T, H, P, T_bias, H_bias, P_bias
        self.n_measurements = 3  # T, H, P
        
        # Initial state
        initial_state = np.array([
            initial_temperature,
            initial_humidity,
            initial_pressure,
            0.0,  # T_bias
            0.0,  # H_bias
            0.0   # P_bias
        ])
        
        # Initial covariance (high uncertainty)
        initial_P = np.diag([
            25.0,    # T variance (5°C std)
            100.0,   # H variance (10% std)
            100.0,   # P variance (10 hPa std)
            1.0,     # T_bias variance
            4.0,     # H_bias variance
            1.0      # P_bias variance
        ])
        
        # Process noise - models how state changes between samples
        # NOTE: This is NOT time-based, it's sample-based
        Q = np.diag([
            0.01 * process_noise_factor,   # T process noise
            0.1 * process_noise_factor,    # H process noise
            0.01 * process_noise_factor,   # P process noise
            0.001 * process_noise_factor,  # T_bias random walk
            0.01 * process_noise_factor,   # H_bias random walk
            0.001 * process_noise_factor   # P_bias random walk
        ])
        
        # Measurement noise
        R = np.diag([
            temperature_noise ** 2,
            humidity_noise ** 2,
            pressure_noise ** 2
        ])
        
        # Create filter
        if adaptive:
            self.ekf = AdaptiveEKF(
                n_states=self.n_states,
                n_measurements=self.n_measurements,
                initial_state=initial_state,
                initial_covariance=initial_P,
                process_noise=Q,
                measurement_noise=R,
                adaptation_rate=0.05,
                window_size=20
            )
        else:
            self.ekf = ExtendedKalmanFilter(
                n_states=self.n_states,
                n_measurements=self.n_measurements,
                initial_state=initial_state,
                initial_covariance=initial_P,
                process_noise=Q,
                measurement_noise=R
            )
        
        # Model parameters
        self.equilibration_rate = np.array([0.1, 0.08, 0.15])  # T, H, P
        self.bias_persistence = 0.99
        self.humidity_temp_coupling = 0.1  # % RH per °C cross-sensitivity
        self.reference_temperature = 25.0
        
        # Ambient estimates (updated adaptively)
        self._ambient_T = initial_temperature
        self._ambient_H = initial_humidity
        self._ambient_P = initial_pressure
        
        # History for trend analysis
        self._history = []
        self._max_history = 100
    
    def _state_transition(self, x: np.ndarray) -> np.ndarray:
        """
        Non-linear state transition function.
        
        Models sensor physics WITHOUT time dependency:
        - Environmental values drift toward ambient
        - Biases follow random walk with decay
        
        This works because consecutive readings from a real sensor
        exhibit these physical relationships regardless of sample timing.
        """
        x = x.flatten()
        x_new = np.zeros(6)
        
        # Environmental states: drift toward ambient estimates
        # This models thermal/diffusive equilibration
        x_new[0] = x[0] + self.equilibration_rate[0] * (self._ambient_T - x[0])
        x_new[1] = x[1] + self.equilibration_rate[1] * (self._ambient_H - x[1])
        x_new[2] = x[2] + self.equilibration_rate[2] * (self._ambient_P - x[2])
        
        # Bias states: random walk with decay
        x_new[3] = self.bias_persistence * x[3]
        x_new[4] = self.bias_persistence * x[4]
        x_new[5] = self.bias_persistence * x[5]
        
        return x_new.reshape(-1, 1)
    
    def _state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of state transition function: F = ∂f/∂x
        
        F = [1-α_T   0       0       0    0    0   ]
            [0       1-α_H   0       0    0    0   ]
            [0       0       1-α_P   0    0    0   ]
            [0       0       0       β    0    0   ]
            [0       0       0       0    β    0   ]
            [0       0       0       0    0    β   ]
        """
        α = self.equilibration_rate
        β = self.bias_persistence
        
        F = np.array([
            [1 - α[0], 0,        0,        0, 0, 0],
            [0,        1 - α[1], 0,        0, 0, 0],
            [0,        0,        1 - α[2], 0, 0, 0],
            [0,        0,        0,        β, 0, 0],
            [0,        0,        0,        0, β, 0],
            [0,        0,        0,        0, 0, β]
        ])
        
        return F
    
    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        Non-linear measurement function.
        
        Maps state to expected sensor readings:
        z_T = T + T_bias
        z_H = H + H_bias + κ*(T - T_ref)  [Temperature compensation]
        z_P = P + P_bias
        
        The humidity-temperature coupling models the physical fact that
        BME sensors' humidity readings are affected by temperature.
        """
        x = x.flatten()
        
        T, H, P = x[0], x[1], x[2]
        T_bias, H_bias, P_bias = x[3], x[4], x[5]
        
        z = np.array([
            T + T_bias,
            H + H_bias + self.humidity_temp_coupling * (T - self.reference_temperature),
            P + P_bias
        ])
        
        return z.reshape(-1, 1)
    
    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of measurement function: H = ∂h/∂x
        
        H = [1   0   0   1   0   0]
            [κ   1   0   0   1   0]  (κ = humidity-temp coupling)
            [0   0   1   0   0   1]
        """
        κ = self.humidity_temp_coupling
        
        H = np.array([
            [1, 0, 0, 1, 0, 0],
            [κ, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1]
        ])
        
        return H
    
    def update(
        self,
        temperature: float,
        humidity: float,
        pressure: float
    ) -> Dict[str, float]:
        """
        Process new BME sensor reading.
        
        This is the main interface for the filter. Simply provide
        raw sensor readings and get filtered estimates.
        
        Args:
            temperature: Raw temperature reading (°C)
            humidity: Raw humidity reading (%)
            pressure: Raw pressure reading (hPa)
        
        Returns:
            Dictionary with filtered estimates and uncertainties:
            {
                'temperature': filtered T,
                'humidity': filtered H,
                'pressure': filtered P,
                'temperature_std': T uncertainty,
                'humidity_std': H uncertainty,
                'pressure_std': P uncertainty,
                'temperature_bias': estimated sensor bias,
                'humidity_bias': estimated sensor bias,
                'pressure_bias': estimated sensor bias
            }
        """
        # Update ambient estimates (exponential moving average)
        α_ambient = 0.02
        self._ambient_T = (1 - α_ambient) * self._ambient_T + α_ambient * temperature
        self._ambient_H = (1 - α_ambient) * self._ambient_H + α_ambient * humidity
        self._ambient_P = (1 - α_ambient) * self._ambient_P + α_ambient * pressure
        
        # Measurement vector
        z = np.array([temperature, humidity, pressure])
        
        # Get Jacobians at current state
        F = self._state_jacobian(self.ekf.x)
        H = self._measurement_jacobian(self.ekf.x)
        
        # Run EKF predict + update
        self.ekf.filter_step(
            measurement=z,
            state_transition_func=self._state_transition,
            measurement_func=self._measurement_function,
            jacobian_F=F,
            jacobian_H=H
        )
        
        # Extract results
        state = self.ekf.get_state()
        uncertainty = self.ekf.get_uncertainty()
        
        # Store in history
        result = {
            'temperature': state[0],
            'humidity': np.clip(state[1], 0, 100),  # Bound humidity to valid range
            'pressure': state[2],
            'temperature_std': uncertainty[0],
            'humidity_std': uncertainty[1],
            'pressure_std': uncertainty[2],
            'temperature_bias': state[3],
            'humidity_bias': state[4],
            'pressure_bias': state[5]
        }
        
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        return result
    
    def get_filtered_values(self) -> Tuple[float, float, float]:
        """Get current filtered (T, H, P) values."""
        state = self.ekf.get_state()
        return state[0], np.clip(state[1], 0, 100), state[2]
    
    def get_sensor_biases(self) -> Tuple[float, float, float]:
        """Get estimated sensor biases (T_bias, H_bias, P_bias)."""
        state = self.ekf.get_state()
        return state[3], state[4], state[5]
    
    def get_uncertainties(self) -> Tuple[float, float, float]:
        """Get current state uncertainties (std) for T, H, P."""
        uncertainty = self.ekf.get_uncertainty()
        return uncertainty[0], uncertainty[1], uncertainty[2]
    
    def get_diagnostics(self) -> Dict:
        """Get filter diagnostics for monitoring."""
        diag = self.ekf.get_diagnostics()
        diag['ambient_estimates'] = {
            'temperature': self._ambient_T,
            'humidity': self._ambient_H,
            'pressure': self._ambient_P
        }
        return diag
    
    def get_history(self) -> List[Dict]:
        """Get filter history for analysis."""
        return self._history.copy()
    
    def reset(
        self,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        pressure: Optional[float] = None
    ):
        """
        Reset filter state.
        
        Args:
            temperature: New initial temperature (or keep current if None)
            humidity: New initial humidity (or keep current if None)
            pressure: New initial pressure (or keep current if None)
        """
        state = self.ekf.get_state()
        
        if temperature is not None:
            state[0] = temperature
            self._ambient_T = temperature
        if humidity is not None:
            state[1] = humidity
            self._ambient_H = humidity
        if pressure is not None:
            state[2] = pressure
            self._ambient_P = pressure
        
        # Reset biases to zero
        state[3:6] = 0
        
        self.ekf.x = state.reshape(-1, 1)
        self.ekf.P = np.diag([25, 100, 100, 1, 4, 1])
        self._history.clear()


class BME680GasEKF(BMESensorEKF):
    """
    Extended EKF for BME680 with Gas Resistance (IAQ).
    
    Extends BMESensorEKF with gas resistance filtering for
    Indoor Air Quality (IAQ) estimation.
    
    Additional State Variables:
    - R_gas: True gas resistance (Ω)
    - R_gas_bias: Gas sensor bias
    - IAQ: Indoor Air Quality index (estimated)
    
    Gas Resistance Model:
    -------------------
    The gas sensor resistance depends on:
    - Air quality (VOCs reduce resistance)
    - Temperature (strong non-linear relationship)
    - Humidity (affects sensor response)
    
    R_gas_measured = R_gas * f(T, H) + bias
    
    where f(T, H) is the temperature-humidity compensation function.
    """
    
    def __init__(
        self,
        initial_gas_resistance: float = 50000.0,
        gas_noise: float = 5000.0,
        **kwargs
    ):
        """
        Initialize BME680 Gas EKF.
        
        Args:
            initial_gas_resistance: Initial gas resistance estimate (Ω)
            gas_noise: Gas resistance measurement noise std (Ω)
            **kwargs: Arguments passed to BMESensorEKF
        """
        super().__init__(**kwargs)
        
        # Extend state vector: add R_gas, R_gas_bias, IAQ_baseline
        self.n_states = 9
        self.n_measurements = 4
        
        # Reinitialize with extended state
        current_state = self.ekf.get_state()
        extended_state = np.zeros(9)
        extended_state[:6] = current_state
        extended_state[6] = initial_gas_resistance
        extended_state[7] = 0.0  # R_gas_bias
        extended_state[8] = initial_gas_resistance  # IAQ baseline
        
        # Extended covariance
        extended_P = np.eye(9)
        extended_P[:6, :6] = self.ekf.P
        extended_P[6, 6] = (initial_gas_resistance * 0.1) ** 2
        extended_P[7, 7] = 1000.0
        extended_P[8, 8] = (initial_gas_resistance * 0.2) ** 2
        
        # Extended process noise
        extended_Q = np.eye(9) * 0.01
        extended_Q[:6, :6] = self.ekf.Q
        extended_Q[6, 6] = 100.0
        extended_Q[7, 7] = 10.0
        extended_Q[8, 8] = 1.0
        
        # Extended measurement noise
        extended_R = np.eye(4)
        extended_R[:3, :3] = self.ekf.R
        extended_R[3, 3] = gas_noise ** 2
        
        # Create new filter with extended dimensions
        self.ekf = ExtendedKalmanFilter(
            n_states=9,
            n_measurements=4,
            initial_state=extended_state,
            initial_covariance=extended_P,
            process_noise=extended_Q,
            measurement_noise=extended_R
        )
        
        # Gas sensor parameters
        self.gas_temp_coefficient = -0.02  # %/°C resistance change
        self.gas_humidity_coefficient = -0.01  # %/%RH resistance change
    
    def _state_transition(self, x: np.ndarray) -> np.ndarray:
        """Extended state transition including gas resistance."""
        x = x.flatten()
        x_new = np.zeros(9)
        
        # Base environmental states (from parent)
        x_new[0] = x[0] + self.equilibration_rate[0] * (self._ambient_T - x[0])
        x_new[1] = x[1] + self.equilibration_rate[1] * (self._ambient_H - x[1])
        x_new[2] = x[2] + self.equilibration_rate[2] * (self._ambient_P - x[2])
        
        # Base biases
        x_new[3] = self.bias_persistence * x[3]
        x_new[4] = self.bias_persistence * x[4]
        x_new[5] = self.bias_persistence * x[5]
        
        # Gas resistance: slow drift toward baseline
        x_new[6] = x[6] + 0.01 * (x[8] - x[6])
        x_new[7] = self.bias_persistence * x[7]  # Gas bias decay
        x_new[8] = x[8]  # Baseline is quasi-static
        
        return x_new.reshape(-1, 1)
    
    def _state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Extended Jacobian for 9-state system."""
        α = self.equilibration_rate
        β = self.bias_persistence
        
        F = np.array([
            [1-α[0], 0,      0,      0, 0, 0, 0,    0, 0   ],
            [0,      1-α[1], 0,      0, 0, 0, 0,    0, 0   ],
            [0,      0,      1-α[2], 0, 0, 0, 0,    0, 0   ],
            [0,      0,      0,      β, 0, 0, 0,    0, 0   ],
            [0,      0,      0,      0, β, 0, 0,    0, 0   ],
            [0,      0,      0,      0, 0, β, 0,    0, 0   ],
            [0,      0,      0,      0, 0, 0, 0.99, 0, 0.01],
            [0,      0,      0,      0, 0, 0, 0,    β, 0   ],
            [0,      0,      0,      0, 0, 0, 0,    0, 1   ]
        ])
        
        return F
    
    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """Extended measurement function including gas."""
        x = x.flatten()
        
        T, H, P = x[0], x[1], x[2]
        T_bias, H_bias, P_bias = x[3], x[4], x[5]
        R_gas, R_gas_bias = x[6], x[7]
        
        # Temperature-humidity compensation for gas sensor
        temp_comp = 1 + self.gas_temp_coefficient * (T - 25)
        humid_comp = 1 + self.gas_humidity_coefficient * (H - 50)
        
        z = np.array([
            T + T_bias,
            H + H_bias + self.humidity_temp_coupling * (T - self.reference_temperature),
            P + P_bias,
            R_gas * temp_comp * humid_comp + R_gas_bias
        ])
        
        return z.reshape(-1, 1)
    
    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Extended measurement Jacobian."""
        x = x.flatten()
        T, H, R_gas = x[0], x[1], x[6]
        κ = self.humidity_temp_coupling
        
        temp_comp = 1 + self.gas_temp_coefficient * (T - 25)
        humid_comp = 1 + self.gas_humidity_coefficient * (H - 50)
        
        # Partial derivatives for gas measurement
        dz4_dT = R_gas * self.gas_temp_coefficient * humid_comp
        dz4_dH = R_gas * temp_comp * self.gas_humidity_coefficient
        dz4_dR = temp_comp * humid_comp
        
        H = np.array([
            [1,     0, 0, 1, 0, 0, 0,     0, 0],
            [κ,     1, 0, 0, 1, 0, 0,     0, 0],
            [0,     0, 1, 0, 0, 1, 0,     0, 0],
            [dz4_dT, dz4_dH, 0, 0, 0, 0, dz4_dR, 1, 0]
        ])
        
        return H
    
    def update_with_gas(
        self,
        temperature: float,
        humidity: float,
        pressure: float,
        gas_resistance: float
    ) -> Dict[str, float]:
        """
        Process BME680 reading including gas resistance.
        
        Args:
            temperature: Raw temperature (°C)
            humidity: Raw humidity (%)
            pressure: Raw pressure (hPa)
            gas_resistance: Raw gas resistance (Ω)
        
        Returns:
            Dictionary with all filtered values including IAQ estimate
        """
        # Update ambient estimates
        α_ambient = 0.02
        self._ambient_T = (1 - α_ambient) * self._ambient_T + α_ambient * temperature
        self._ambient_H = (1 - α_ambient) * self._ambient_H + α_ambient * humidity
        self._ambient_P = (1 - α_ambient) * self._ambient_P + α_ambient * pressure
        
        # Measurement vector
        z = np.array([temperature, humidity, pressure, gas_resistance])
        
        # Get Jacobians
        F = self._state_jacobian(self.ekf.x)
        H = self._measurement_jacobian(self.ekf.x)
        
        # Run EKF
        self.ekf.filter_step(
            measurement=z,
            state_transition_func=self._state_transition,
            measurement_func=self._measurement_function,
            jacobian_F=F,
            jacobian_H=H
        )
        
        state = self.ekf.get_state()
        uncertainty = self.ekf.get_uncertainty()
        
        # Estimate IAQ from gas resistance ratio
        baseline = state[8]
        current = state[6]
        iaq_ratio = current / baseline if baseline > 0 else 1.0
        
        # Simple IAQ mapping (0-500 scale, 100 = good)
        iaq = 100 * (2 - iaq_ratio)
        iaq = np.clip(iaq, 0, 500)
        
        result = {
            'temperature': state[0],
            'humidity': np.clip(state[1], 0, 100),
            'pressure': state[2],
            'gas_resistance': state[6],
            'iaq': iaq,
            'temperature_std': uncertainty[0],
            'humidity_std': uncertainty[1],
            'pressure_std': uncertainty[2],
            'gas_resistance_std': uncertainty[6],
            'temperature_bias': state[3],
            'humidity_bias': state[4],
            'pressure_bias': state[5],
            'gas_bias': state[7],
            'gas_baseline': state[8]
        }
        
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        return result
