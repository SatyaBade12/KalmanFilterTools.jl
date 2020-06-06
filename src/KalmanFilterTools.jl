@doc raw"""
Module KalmanFilterTools provides algorithms for computing the
Kalman filter, the Kalman smoother and the log-likelihood of a state
space model using the Kalman filter.

The state space is given by
```
    y_t = Z_t α_t + ϵ_t
    α_{t+1} = T α_t + Rη_t
```
"""
module KalmanFilterTools

"""
State space specification:
    y_t = Z*a_t + epsilon_t
    a_{t+1} = T*a_t- + R eta_t
    E(epsilon_t epsilon_t') = H
    E(eta_t eta_t') = Q
"""

using LinearAlgebra
using LinearAlgebra.BLAS

export KalmanLikelihoodWs, FastKalmanLikelihoodWs, DiffuseKalmanLikelihoodWs
export DiffuseKalmanFilterWs
export KalmanSmootherWs, kalman_likelihood, kalman_likelihood_monitored
export fast_kalman_likelihood, diffuse_kalman_likelihood, kalman_filter!
export kalman_smoother!

abstract type KalmanWs{T, U} end

include("kalman_base.jl")
include("univariate_step.jl")
include("kalman_likelihood.jl")
include("kalman_filter.jl")
include("kalman_smoother.jl")

end #module
