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
export KalmanFilterWs, kalman_likelihood, kalman_likelihood_monitored
export DiffuseKalmanFilterWs
export fast_kalman_likelihood, diffuse_kalman_likelihood
export kalman_filter!, diffuse_kalman_filter!
export KalmanSmootherWs, DiffuseKalmanSmootherWs, kalman_smoother!
export diffuse_kalman_smoother!

abstract type KalmanWs{T, U} end

include("kalman_base.jl")
include("kalman_likelihood.jl")
include("kalman_filter.jl")
include("kalman_smoother.jl")
include("univariate_step.jl")

import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
  include("../test/runtests.jl")
end

end #module
