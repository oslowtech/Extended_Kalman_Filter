"""
Extended Kalman Filter (EKF) Core Implementation
================================================

This module provides a generic Extended Kalman Filter implementation
designed for non-linear systems where standard Kalman filters fail.

Key Difference from Standard Kalman Filter:
-------------------------------------------
Standard Kalman Filter assumes:
    x(k+1) = A * x(k) + B * u(k) + w(k)  [LINEAR state transition]
    z(k) = H * x(k) + v(k)                [LINEAR measurement]

This fails for BME sensors because:
1. Sensor readings are NOT linearly related to true physical values
2. Standard KF requires TIME as a reference (Δt for prediction)
3. Environmental sensors exhibit non-linear drift and hysteresis

Extended Kalman Filter solves this by:
1. Using non-linear state transition: x(k+1) = f(x(k), u(k)) + w(k)
2. Using non-linear measurement model: z(k) = h(x(k)) + v(k)
3. Linearizing via Jacobian matrices at each step
4. Operating on sensor VALUE relationships, NOT time progression

Author: BME Sensor Fusion System
License: MIT
"""

import numpy as np
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod


class ExtendedKalmanFilter:
    """
    Generic Extended Kalman Filter Implementation
    
    This EKF operates WITHOUT requiring time as a reference.
    Instead, it uses the relationship between measurements
    and state estimates to perform filtering.
    
    Mathematical Foundation:
    -----------------------
    
    STATE SPACE MODEL (Non-linear):
        State Prediction:     x̂(k|k-1) = f(x̂(k-1|k-1))
        Measurement Model:    ẑ(k) = h(x̂(k|k-1))
    
    LINEARIZATION (via Jacobians):
        F = ∂f/∂x |x=x̂(k-1|k-1)    [State transition Jacobian]
        H = ∂h/∂x |x=x̂(k|k-1)      [Measurement Jacobian]
    
    PREDICTION STEP:
        x̂(k|k-1) = f(x̂(k-1|k-1))
        P(k|k-1) = F * P(k-1|k-1) * F^T + Q
    
    UPDATE STEP:
        y(k) = z(k) - h(x̂(k|k-1))           [Innovation/Residual]
        S(k) = H * P(k|k-1) * H^T + R        [Innovation Covariance]
        K(k) = P(k|k-1) * H^T * S(k)^(-1)    [Kalman Gain]
        x̂(k|k) = x̂(k|k-1) + K(k) * y(k)     [State Update]
        P(k|k) = (I - K(k) * H) * P(k|k-1)   [Covariance Update]
    
    Why No Time Reference is Needed:
    --------------------------------
    Traditional filters use: x(k+1) = x(k) + v*Δt
    
    This EKF uses: x(k+1) = f(x(k))
    
    where f() models the relationship between consecutive states
    based on SENSOR PHYSICS, not temporal evolution:
    - Temperature inertia (thermal mass effects)
    - Humidity equilibration dynamics
    - Pressure stability characteristics
    
    Attributes:
        n_states (int): Number of state variables
        n_measurements (int): Number of measurement variables
        x (np.ndarray): State estimate vector [n_states, 1]
        P (np.ndarray): State covariance matrix [n_states, n_states]
        Q (np.ndarray): Process noise covariance [n_states, n_states]
        R (np.ndarray): Measurement noise covariance [n_measurements, n_measurements]
    """
    
    def __init__(
        self,
        n_states: int,
        n_measurements: int,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None
    ):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            n_states: Dimension of state vector
            n_measurements: Dimension of measurement vector
            initial_state: Initial state estimate (defaults to zeros)
            initial_covariance: Initial uncertainty (defaults to identity)
            process_noise: Process noise Q (defaults to small identity)
            measurement_noise: Measurement noise R (defaults to identity)
        """
        self.n_states = n_states
        self.n_measurements = n_measurements
        
        # State estimate
        if initial_state is not None:
            self.x = np.array(initial_state).reshape(n_states, 1)
        else:
            self.x = np.zeros((n_states, 1))
        
        # State covariance (uncertainty)
        if initial_covariance is not None:
            self.P = np.array(initial_covariance)
        else:
            self.P = np.eye(n_states) * 1000.0  # High initial uncertainty
        
        # Process noise covariance
        if process_noise is not None:
            self.Q = np.array(process_noise)
        else:
            self.Q = np.eye(n_states) * 0.01
        
        # Measurement noise covariance
        if measurement_noise is not None:
            self.R = np.array(measurement_noise)
        else:
            self.R = np.eye(n_measurements) * 1.0
        
        # Identity matrix for updates
        self._I = np.eye(n_states)
        
        # Innovation sequence (for diagnostics)
        self._innovation = None
        self._innovation_covariance = None
        self._kalman_gain = None
    
    def predict(
        self,
        state_transition_func: Callable[[np.ndarray], np.ndarray],
        jacobian_F: np.ndarray
    ) -> np.ndarray:
        """
        Prediction Step - Project state ahead WITHOUT time reference.
        
        This step predicts the next state based on the non-linear
        state transition function f(x), NOT based on time evolution.
        
        Mathematical Operation:
            x̂(k|k-1) = f(x̂(k-1|k-1))
            P(k|k-1) = F * P(k-1|k-1) * F^T + Q
        
        Args:
            state_transition_func: Function f(x) -> x_predicted
                                   Models sensor physics, NOT time!
            jacobian_F: Jacobian matrix ∂f/∂x evaluated at current state
                       Shape: [n_states, n_states]
        
        Returns:
            Predicted state estimate x̂(k|k-1)
        """
        # State prediction using non-linear function
        self.x = state_transition_func(self.x)
        
        # Covariance prediction using linearized model
        self.P = jacobian_F @ self.P @ jacobian_F.T + self.Q
        
        return self.x.copy()
    
    def update(
        self,
        measurement: np.ndarray,
        measurement_func: Callable[[np.ndarray], np.ndarray],
        jacobian_H: np.ndarray
    ) -> np.ndarray:
        """
        Update Step - Incorporate measurement to correct prediction.
        
        This step uses the actual sensor reading to correct the
        predicted state estimate using statistical fusion.
        
        Mathematical Operation:
            y(k) = z(k) - h(x̂(k|k-1))           [Innovation]
            S(k) = H * P(k|k-1) * H^T + R        [Innovation Covariance]
            K(k) = P(k|k-1) * H^T * S(k)^(-1)    [Kalman Gain]
            x̂(k|k) = x̂(k|k-1) + K(k) * y(k)     [State Update]
            P(k|k) = (I - K(k) * H) * P(k|k-1)   [Covariance Update]
        
        Args:
            measurement: Actual sensor measurement z(k)
                        Shape: [n_measurements, 1] or [n_measurements,]
            measurement_func: Function h(x) -> z_expected
                             Maps state to expected measurement
            jacobian_H: Jacobian matrix ∂h/∂x evaluated at predicted state
                       Shape: [n_measurements, n_states]
        
        Returns:
            Updated state estimate x̂(k|k)
        """
        z = np.array(measurement).reshape(self.n_measurements, 1)
        
        # Predicted measurement from current state estimate
        z_predicted = measurement_func(self.x)
        
        # Innovation (measurement residual)
        y = z - z_predicted
        self._innovation = y.copy()
        
        # Innovation covariance
        S = jacobian_H @ self.P @ jacobian_H.T + self.R
        self._innovation_covariance = S.copy()
        
        # Kalman gain - determines trust between prediction and measurement
        # High K = trust measurement more
        # Low K = trust prediction more
        K = self.P @ jacobian_H.T @ np.linalg.inv(S)
        self._kalman_gain = K.copy()
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        IKH = self._I - K @ jacobian_H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        
        return self.x.copy()
    
    def filter_step(
        self,
        measurement: np.ndarray,
        state_transition_func: Callable[[np.ndarray], np.ndarray],
        measurement_func: Callable[[np.ndarray], np.ndarray],
        jacobian_F: np.ndarray,
        jacobian_H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete filter cycle: Predict + Update in one step.
        
        Args:
            measurement: Actual sensor measurement z(k)
            state_transition_func: Non-linear state transition f(x)
            measurement_func: Non-linear measurement function h(x)
            jacobian_F: State transition Jacobian ∂f/∂x
            jacobian_H: Measurement Jacobian ∂h/∂x
        
        Returns:
            Tuple of (state_estimate, state_covariance)
        """
        self.predict(state_transition_func, jacobian_F)
        self.update(measurement, measurement_func, jacobian_H)
        return self.x.copy(), self.P.copy()
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x.flatten()
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance (uncertainty)."""
        return self.P.copy()
    
    def get_uncertainty(self) -> np.ndarray:
        """Get state uncertainty as standard deviations."""
        return np.sqrt(np.diag(self.P))
    
    def get_diagnostics(self) -> dict:
        """
        Get filter diagnostics for performance monitoring.
        
        Returns:
            Dictionary containing:
            - innovation: Latest measurement residual
            - innovation_covariance: Innovation uncertainty
            - kalman_gain: Latest Kalman gain matrix
            - normalized_innovation_squared: NIS for consistency check
        """
        if self._innovation is None:
            return {}
        
        # Normalized Innovation Squared (NIS) - should follow chi-squared distribution
        NIS = float(self._innovation.T @ np.linalg.inv(self._innovation_covariance) @ self._innovation)
        
        return {
            'innovation': self._innovation.flatten(),
            'innovation_covariance': self._innovation_covariance,
            'kalman_gain': self._kalman_gain,
            'normalized_innovation_squared': NIS
        }


class AdaptiveEKF(ExtendedKalmanFilter):
    """
    Adaptive Extended Kalman Filter with automatic noise estimation.
    
    This variant automatically adjusts Q and R matrices based on
    innovation sequence analysis, making it robust to changing
    sensor characteristics without manual tuning.
    
    Adaptation Mechanism:
    --------------------
    The filter monitors the innovation sequence y(k) = z(k) - h(x̂(k|k-1))
    
    For a properly tuned filter:
        E[y(k)] = 0  (zero mean)
        E[y(k) * y(k)^T] = S(k)  (matches innovation covariance)
    
    If innovations are too large: Increase R (trust measurements less)
    If innovations are too small: Decrease R (trust measurements more)
    
    This self-tuning eliminates the need for precise noise characterization.
    """
    
    def __init__(
        self,
        n_states: int,
        n_measurements: int,
        adaptation_rate: float = 0.1,
        window_size: int = 10,
        **kwargs
    ):
        """
        Initialize Adaptive EKF.
        
        Args:
            n_states: Dimension of state vector
            n_measurements: Dimension of measurement vector
            adaptation_rate: Learning rate for noise adaptation (0 to 1)
            window_size: Number of samples for adaptation statistics
            **kwargs: Additional arguments passed to base EKF
        """
        super().__init__(n_states, n_measurements, **kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # Innovation history for adaptation
        self._innovation_history = []
    
    def update(
        self,
        measurement: np.ndarray,
        measurement_func: Callable[[np.ndarray], np.ndarray],
        jacobian_H: np.ndarray
    ) -> np.ndarray:
        """
        Update with adaptive noise estimation.
        
        Extends base update with innovation-based R adaptation.
        """
        result = super().update(measurement, measurement_func, jacobian_H)
        
        # Store innovation for adaptation
        self._innovation_history.append(self._innovation.flatten())
        if len(self._innovation_history) > self.window_size:
            self._innovation_history.pop(0)
        
        # Adapt measurement noise if enough samples
        if len(self._innovation_history) >= self.window_size:
            self._adapt_measurement_noise(jacobian_H)
        
        return result
    
    def _adapt_measurement_noise(self, jacobian_H: np.ndarray):
        """
        Adapt R matrix based on innovation statistics.
        
        Uses covariance matching: The sample covariance of innovations
        should match the theoretical innovation covariance S = HPH^T + R.
        
        Solving for R: R_new = C_y - H * P * H^T
        where C_y is the sample innovation covariance.
        """
        # Compute sample innovation covariance
        innovations = np.array(self._innovation_history)
        sample_cov = np.cov(innovations.T)
        
        if sample_cov.ndim == 0:
            sample_cov = np.array([[sample_cov]])
        
        # Expected innovation covariance from filter
        expected_cov = jacobian_H @ self.P @ jacobian_H.T + self.R
        
        # Adaptation: blend current R with estimated R
        R_estimated = sample_cov - jacobian_H @ self.P @ jacobian_H.T
        
        # Ensure positive definiteness
        R_estimated = (R_estimated + R_estimated.T) / 2
        eigvals = np.linalg.eigvalsh(R_estimated)
        if np.min(eigvals) < 0:
            R_estimated += (abs(np.min(eigvals)) + 1e-6) * np.eye(self.n_measurements)
        
        # Blend with current R
        self.R = (1 - self.adaptation_rate) * self.R + self.adaptation_rate * R_estimated


def compute_numerical_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    epsilon: float = 1e-7
) -> np.ndarray:
    """
    Compute Jacobian matrix numerically using finite differences.
    
    This is useful when analytical Jacobian is difficult to derive.
    Uses central difference for better accuracy.
    
    J[i,j] = ∂f_i/∂x_j ≈ (f_i(x + ε*e_j) - f_i(x - ε*e_j)) / (2ε)
    
    Args:
        func: Function to differentiate
        x: Point at which to evaluate Jacobian
        epsilon: Perturbation size
    
    Returns:
        Jacobian matrix [n_outputs, n_inputs]
    """
    x = np.array(x).flatten()
    n = len(x)
    
    f0 = func(x.reshape(-1, 1)).flatten()
    m = len(f0)
    
    J = np.zeros((m, n))
    
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += epsilon
        x_minus[j] -= epsilon
        
        f_plus = func(x_plus.reshape(-1, 1)).flatten()
        f_minus = func(x_minus.reshape(-1, 1)).flatten()
        
        J[:, j] = (f_plus - f_minus) / (2 * epsilon)
    
    return J
