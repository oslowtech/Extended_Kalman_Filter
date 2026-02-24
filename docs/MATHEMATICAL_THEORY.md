# Extended Kalman Filter: Mathematical Theory and Derivation

## Table of Contents
1. [Introduction: Why Standard Kalman Filter Fails](#1-introduction-why-standard-kalman-filter-fails)
2. [Extended Kalman Filter Theory](#2-extended-kalman-filter-theory)
3. [Mathematical Derivation](#3-mathematical-derivation)
4. [Value-Based Filtering (No Time Reference)](#4-value-based-filtering-no-time-reference)
5. [BME Sensor Specific Formulation](#5-bme-sensor-specific-formulation)
6. [Jacobian Derivations](#6-jacobian-derivations)
7. [Convergence and Stability](#7-convergence-and-stability)
8. [Practical Considerations](#8-practical-considerations)

---

## 1. Introduction: Why Standard Kalman Filter Fails

### 1.1 Standard Kalman Filter Assumptions

The Standard Kalman Filter (SKF) operates on a linear state-space model:

**State Transition (Prediction):**
$$x_{k+1} = A \cdot x_k + B \cdot u_k + w_k$$

**Measurement Model:**
$$z_k = H \cdot x_k + v_k$$

Where:
- $x_k$ = State vector at time step $k$
- $A$ = State transition matrix (LINEAR)
- $B$ = Control input matrix
- $u_k$ = Control input
- $H$ = Measurement matrix (LINEAR)
- $w_k \sim \mathcal{N}(0, Q)$ = Process noise
- $v_k \sim \mathcal{N}(0, R)$ = Measurement noise

### 1.2 Why This Fails for BME Sensors

#### Problem 1: Time-Based State Evolution

Standard KF typically models state evolution as:

$$x_{k+1} = x_k + \dot{x}_k \cdot \Delta t$$

For position tracking: $position_{k+1} = position_k + velocity \cdot \Delta t$

**This makes no sense for environmental sensors:**
- Temperature doesn't have a "velocity"
- Humidity doesn't follow: $H_{k+1} = H_k + \frac{dH}{dt} \cdot \Delta t$
- There's no physical basis for time-based prediction

#### Problem 2: Non-Linear Sensor Response

BME sensors exhibit non-linear behaviors:

**Humidity-Temperature Coupling:**
$$z_H = H_{true} + \kappa \cdot (T - T_{ref})$$

Where humidity readings change with temperature even if actual humidity is constant.

**Pressure-Temperature Relationship:**
$$P_{true} = P_{measured} \cdot \left(1 + \alpha(T - T_{cal})\right)$$

These relationships are **multiplicative and non-linear**, violating the linear $z = H \cdot x$ assumption.

#### Problem 3: Sensor Drift and Hysteresis

BME sensors show:
- Long-term drift that isn't time-predictable
- Hysteresis (path-dependent readings)
- Non-linear response to rapid changes

A linear prediction model cannot capture these effects.

---

## 2. Extended Kalman Filter Theory

### 2.1 Non-Linear State-Space Model

The EKF operates on non-linear functions:

**State Transition:**
$$\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1})$$

**Measurement Model:**
$$\hat{z}_k = h(\hat{x}_{k|k-1})$$

Where $f(\cdot)$ and $h(\cdot)$ are **non-linear functions**.

### 2.2 Key Insight: Linearization via Taylor Expansion

At each step, we linearize around the current estimate:

**First-Order Taylor Expansion:**
$$f(x) \approx f(\hat{x}) + F \cdot (x - \hat{x})$$
$$h(x) \approx h(\hat{x}) + H \cdot (x - \hat{x})$$

Where the Jacobian matrices are:

$$F = \frac{\partial f}{\partial x}\bigg|_{x=\hat{x}_{k-1|k-1}}$$

$$H = \frac{\partial h}{\partial x}\bigg|_{x=\hat{x}_{k|k-1}}$$

### 2.3 EKF Algorithm

**PREDICTION STEP:**

1. Predict state:
$$\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1})$$

2. Predict covariance:
$$P_{k|k-1} = F \cdot P_{k-1|k-1} \cdot F^T + Q$$

**UPDATE STEP:**

1. Compute innovation (measurement residual):
$$y_k = z_k - h(\hat{x}_{k|k-1})$$

2. Compute innovation covariance:
$$S_k = H \cdot P_{k|k-1} \cdot H^T + R$$

3. Compute Kalman gain:
$$K_k = P_{k|k-1} \cdot H^T \cdot S_k^{-1}$$

4. Update state:
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \cdot y_k$$

5. Update covariance:
$$P_{k|k} = (I - K_k \cdot H) \cdot P_{k|k-1}$$

---

## 3. Mathematical Derivation

### 3.1 Minimum Mean Square Error (MMSE) Estimator

The Kalman Filter is derived as the MMSE estimator:

$$\hat{x} = \arg\min_{\hat{x}} E\left[(x - \hat{x})^T (x - \hat{x})\right]$$

For Gaussian distributions, this equals the Maximum A Posteriori (MAP) estimate.

### 3.2 Bayesian Interpretation

**Prior Distribution (Prediction):**
$$p(x_k | z_{1:k-1}) = \mathcal{N}(\hat{x}_{k|k-1}, P_{k|k-1})$$

**Likelihood (Measurement Model):**
$$p(z_k | x_k) = \mathcal{N}(h(x_k), R)$$

**Posterior Distribution (Update):**
$$p(x_k | z_{1:k}) \propto p(z_k | x_k) \cdot p(x_k | z_{1:k-1})$$

The EKF approximates this by assuming the posterior is Gaussian.

### 3.3 Kalman Gain Derivation

The Kalman gain minimizes the posterior covariance trace:

$$K_k = \arg\min_K \text{tr}(P_{k|k})$$

Taking the derivative and setting to zero:

$$\frac{\partial}{\partial K} \text{tr}\left[(I - K H) P_{k|k-1} (I - K H)^T + K R K^T\right] = 0$$

Solving yields:
$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$

### 3.4 Covariance Update Derivation

Starting from:
$$P_{k|k} = E\left[(\hat{x}_{k|k} - x_k)(\hat{x}_{k|k} - x_k)^T\right]$$

Substituting the update equation:
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K(z - H\hat{x}_{k|k-1})$$

And $z = Hx + v$:
$$P_{k|k} = (I - KH)P_{k|k-1}(I - KH)^T + KRK^T$$

This is the Joseph form, which is numerically stable.

---

## 4. Value-Based Filtering (No Time Reference)

### 4.1 The Fundamental Shift

**Traditional (Time-Based) Filtering:**
- State evolves according to: $x(t + \Delta t) = x(t) + f(\dot{x}, \Delta t)$
- Requires known time intervals
- Models temporal dynamics

**Value-Based Filtering:**
- State evolves according to: $x_{k+1} = f(x_k, sensor\_physics)$
- Requires only sequential measurements
- Models sensor/physical relationships

### 4.2 Why This Works

For BME sensors, consecutive readings are related by **physical laws**, not time:

#### Thermal Equilibration Model

$$T_{k+1} = T_k + \alpha_T \cdot (T_{ambient} - T_k)$$

This models the fact that sensor temperature drifts toward ambient.
- $\alpha_T$ is an equilibration rate (related to thermal mass)
- This works **per sample**, regardless of sampling rate
- The relationship is based on **physics**, not elapsed time

#### Sensor Response Model

$$T_{measured} = T_{true} + bias + \epsilon$$

The measurement model relates true values to sensor readings.
This is independent of when the measurement was taken.

### 4.3 Mathematical Justification

Consider a stationary stochastic process where:
$$E[x_k] = \mu \quad \text{(constant mean)}$$
$$\text{Cov}(x_k, x_{k+1}) = \rho \cdot \sigma^2 \quad \text{(autocorrelation)}$$

The optimal linear predictor is:
$$\hat{x}_{k+1|k} = \mu + \rho \cdot (x_k - \mu)$$

This has the form:
$$\hat{x}_{k+1|k} = (1-\rho) \cdot \mu + \rho \cdot x_k$$

Which matches our equilibration model with:
- $\alpha = 1 - \rho$
- $T_{ambient} = \mu$

**No time reference needed** - the relationship is sample-to-sample.

### 4.4 Advantages of Value-Based Filtering

1. **No Timing Assumptions**: Works with irregular sampling
2. **Robust to Clock Drift**: No timestamp dependencies
3. **Simpler Implementation**: No time bookkeeping
4. **Physically Meaningful**: Parameters relate to sensor physics
5. **Adaptive**: Filter learns the actual sensor behavior

---

## 5. BME Sensor Specific Formulation

### 5.1 State Vector Definition

$$x = \begin{bmatrix} T \\ H \\ P \\ b_T \\ b_H \\ b_P \end{bmatrix}$$

Where:
- $T$: True temperature (°C)
- $H$: True relative humidity (%)
- $P$: True barometric pressure (hPa)
- $b_T, b_H, b_P$: Sensor bias terms

### 5.2 State Transition Function

$$f(x) = \begin{bmatrix} 
T + \alpha_T(T_{amb} - T) \\
H + \alpha_H(H_{amb} - H) \\
P + \alpha_P(P_{amb} - P) \\
\beta \cdot b_T \\
\beta \cdot b_H \\
\beta \cdot b_P
\end{bmatrix}$$

**Parameters:**
- $\alpha_T, \alpha_H, \alpha_P \in (0, 1)$: Equilibration rates
- $\beta \in (0, 1)$: Bias persistence factor
- $T_{amb}, H_{amb}, P_{amb}$: Ambient estimates (updated adaptively)

### 5.3 Measurement Function

$$h(x) = \begin{bmatrix} 
T + b_T \\
H + b_H + \kappa(T - T_{ref}) \\
P + b_P
\end{bmatrix}$$

**Parameters:**
- $\kappa$: Humidity-temperature cross-sensitivity coefficient
- $T_{ref}$: Reference temperature (typically 25°C)

The cross-term $\kappa(T - T_{ref})$ models the fact that humidity sensors produce different readings at different temperatures, even for the same actual humidity.

### 5.4 State Transition Jacobian

$$F = \frac{\partial f}{\partial x} = \begin{bmatrix}
1-\alpha_T & 0 & 0 & 0 & 0 & 0 \\
0 & 1-\alpha_H & 0 & 0 & 0 & 0 \\
0 & 0 & 1-\alpha_P & 0 & 0 & 0 \\
0 & 0 & 0 & \beta & 0 & 0 \\
0 & 0 & 0 & 0 & \beta & 0 \\
0 & 0 & 0 & 0 & 0 & \beta
\end{bmatrix}$$

### 5.5 Measurement Jacobian

$$H = \frac{\partial h}{\partial x} = \begin{bmatrix}
1 & 0 & 0 & 1 & 0 & 0 \\
\kappa & 1 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 1
\end{bmatrix}$$

### 5.6 Noise Covariance Matrices

**Process Noise (Q):**
$$Q = \begin{bmatrix}
\sigma^2_{T,proc} & 0 & 0 & 0 & 0 & 0 \\
0 & \sigma^2_{H,proc} & 0 & 0 & 0 & 0 \\
0 & 0 & \sigma^2_{P,proc} & 0 & 0 & 0 \\
0 & 0 & 0 & \sigma^2_{bT} & 0 & 0 \\
0 & 0 & 0 & 0 & \sigma^2_{bH} & 0 \\
0 & 0 & 0 & 0 & 0 & \sigma^2_{bP}
\end{bmatrix}$$

**Measurement Noise (R):**
$$R = \begin{bmatrix}
\sigma^2_T & 0 & 0 \\
0 & \sigma^2_H & 0 \\
0 & 0 & \sigma^2_P
\end{bmatrix}$$

---

## 6. Jacobian Derivations

### 6.1 State Transition Jacobian Derivation

For the state transition:
$$f_1 = T + \alpha_T(T_{amb} - T) = (1-\alpha_T)T + \alpha_T \cdot T_{amb}$$

Computing partial derivatives:
$$\frac{\partial f_1}{\partial T} = 1 - \alpha_T$$
$$\frac{\partial f_1}{\partial H} = 0$$
$$\frac{\partial f_1}{\partial b_T} = 0$$

Similarly for all other states, yielding the diagonal structure.

### 6.2 Measurement Jacobian Derivation

For the humidity measurement:
$$h_2 = H + b_H + \kappa(T - T_{ref})$$

Computing partial derivatives:
$$\frac{\partial h_2}{\partial T} = \kappa$$
$$\frac{\partial h_2}{\partial H} = 1$$
$$\frac{\partial h_2}{\partial b_H} = 1$$

This yields the non-zero cross-term in the measurement Jacobian.

### 6.3 Numerical Jacobian Computation

For complex non-linear functions, compute numerically:

$$\frac{\partial f_i}{\partial x_j} \approx \frac{f_i(x + \epsilon e_j) - f_i(x - \epsilon e_j)}{2\epsilon}$$

Where $e_j$ is the unit vector in direction $j$ and $\epsilon \approx 10^{-7}$.

---

## 7. Convergence and Stability

### 7.1 Observability

The system is observable if the observability matrix has full rank:

$$\mathcal{O} = \begin{bmatrix} H \\ HF \\ HF^2 \\ \vdots \\ HF^{n-1} \end{bmatrix}$$

For the BME filter, $\text{rank}(\mathcal{O}) = 6 = n$, so the system is observable.

### 7.2 Stability Conditions

The EKF is stable if:
1. $(F, H)$ is observable
2. $(F, \sqrt{Q})$ is controllable
3. The linearization error is bounded

### 7.3 Covariance Boundedness

The covariance matrix $P$ remains bounded if:
$$\lambda_{min}(Q) > 0 \quad \text{and} \quad \lambda_{min}(R) > 0$$

This ensures the filter doesn't become overconfident.

### 7.4 Consistency

The filter is consistent if:
$$E[y_k y_k^T] \approx S_k$$

The Normalized Innovation Squared (NIS) should follow:
$$\gamma_k = y_k^T S_k^{-1} y_k \sim \chi^2_m$$

Where $m$ is the measurement dimension.

---

## 8. Practical Considerations

### 8.1 Initial Covariance Selection

Choose large initial $P_0$ to reflect uncertainty:
- Too small: Filter ignores initial measurements
- Too large: Slow convergence

Recommendation: Set diagonal elements to square of expected maximum error.

### 8.2 Process Noise Tuning

Process noise $Q$ represents model uncertainty:
- Too small: Filter trusts model too much, becomes sluggish
- Too large: Filter becomes noisy, tracks measurement noise

**Adaptive Approach**: Use innovation sequence to auto-tune.

### 8.3 Measurement Noise Estimation

From sensor specifications:
- BME280 Temperature: ±0.5°C → $\sigma_T = 0.5$
- BME280 Humidity: ±3% → $\sigma_H = 3$
- BME280 Pressure: ±1 hPa → $\sigma_P = 1$

### 8.4 Handling Outliers

Outlier detection using innovation:
$$|y_k| > \gamma \cdot \sqrt{S_k}$$

Where $\gamma \approx 3$ for 3-sigma detection.

For detected outliers:
1. Increase measurement noise temporarily
2. Skip update step
3. Use robust M-estimator

### 8.5 Numerical Stability

Use Joseph form for covariance update:
$$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T$$

This maintains symmetry and positive definiteness.

### 8.6 Real-Time Performance

Computational complexity per update:
- Prediction: $O(n^2)$
- Update: $O(n^2 m + m^3)$

Where $n$ = state dimension, $m$ = measurement dimension.

For BME filter ($n=6$, $m=3$): ~100 floating-point operations per update.

---

## Summary

The Extended Kalman Filter for BME sensors:

1. **Solves the time-reference problem** by using value-based state transition models based on sensor physics
2. **Handles non-linearities** through Jacobian-based linearization at each step
3. **Estimates sensor biases** as state variables for automatic calibration
4. **Models cross-sensitivities** (like temperature effect on humidity readings)
5. **Provides uncertainty estimates** for confidence bounds on filtered values

This approach produces stable, accurate filtered outputs without requiring time intervals or complex temporal models.
