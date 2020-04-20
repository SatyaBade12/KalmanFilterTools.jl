# KalmanFilterTools.jl

WORK IN PROGRESS

KalmanFilterTools provides efficient code to perform various computations pertaining to state space models and the Kalman Filter, such as the Kalman filter proper, the Kalman smoother or computing the log likelihood for the model.

Because such operations are very often computed in an iterative manner, all operations are computed /in place/. One function allocate the necessary workspace and another function performs the computations.

## Installation

KalmanFilterTools.jl is available on GitLab:

```
(v1.4) pkg> add https://git.dynare.org/julia-packages/kalmanfiltertools.jl.git
```

## Julia version

KalmanFilterTools requires Julia version >= 1.4

## State Space model
KalmanFilterTools handles state space models of the following form:

```
  y_t = Z a_t + \epsilon_t
  a_{t+1} = Ta_t + R\eta_t

  \epsilon_t \sim N(0,H)
  \eta_t \sim N(0,Q)
```

  ``y_t``: observation vector ny x 1
  ``a_t``: state vector ns x 1
  ``\epsilon_t``: measurement error vector ny x 1
  ``\eta_t``: shocks vector np x 1
  ``Z``: ny x ns matrix
  ``T``: ns x ns matrix
  ``R``: ns x np matrix
  ``H``: ny x ny covariance matrix
  ``Q``: ns x ns covariance matrix

## Example

Computing the log likelihood

```
 using KalmanFilterTools

 data = ....
 Z = ...
 T = ...
 R = ...
 Q = ...
 a = ...
 P = ...
 
 ny, ns = size(Z)
 np = size(R, 2)
 nobs = size(data,2)
 first_obs = 1
 last_obs = nobs
 presample = 0
 
 kalman_ws = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)

 llk = kalman_likelihood(data, Z, H, T, R, Q, a, P, first_obs, last_obs, presample, kalman_ws)
``` 
