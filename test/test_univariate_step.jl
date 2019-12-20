using LinearAlgebra
using LinearAlgebra.BLAS
using KalmanFilterTools
using Test

ny = 3
ns = 10
np   = 2
nobs = 50

ws = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)

y = randn(ny)
ystar = similar(y)
Z = randn(ny, ns)
Zstar = similar(Z)
H = randn(ny, ny)
H = H'*H
cholH = copy(H)
LAPACK.potrf!('L', H)
detLTcholH = KalmanFilterTools.transformed_measurement!(ystar, Zstar, y, Z, cholH)
@test y ≈ LowerTriangular(cholH)*ystar
@test detLTcholH ≈ det(LowerTriangular(cholH))

nobs = 1
ws = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)
Y = randn(ny, nobs+1)
t = 1
ystar = similar(y)
Z = randn(ny, ns)
Zstar = similar(Z)
H = randn(ny, ny)
H = H'*H
#H = zeros(ny, ny)
copy!(ws.cholH, H)
LAPACK.potrf!('L', ws.cholH)
T = randn(ns, ns)
Q = randn(np, np)
Q = Q'*Q
R = randn(ns, np)
RQR = R*Q*R'
a = randn(ns)
P = randn(ns, ns)
P = P'*P
kalman_tol = eps()^(2/3)

a0 = copy(a)
P0 = copy(P)
lik0 = KalmanFilterTools.univariate_step!(Y, t, Z, H, T, RQR, a0, P0, kalman_tol, ws)
a1 = copy(a0)
P1 = copy(P0)

#=
ws1 = KalmanLikelihoodWs{Float64, Integer}(1, ns, np, 3)
a0 = copy(a)
aa0 = cat(a0, zeros(ns), zeros(ns), zeros(ns); dims=3)
P0 = copy(P)
PP0 = cat(P0, zeros(ns,ns), zeros(ns,ns), zeros(ns,ns); dims=3)
ZZ = cat(Z[1,:]', Z[2,:]', Z[3,:]'; dims=3)
TT = I(ns) + zeros(ns, ns)
lik1a = KalmanFilterTools.kalman_filter!(Y[:,1]', zeros(3), ZZ, H, zeros(ns), TT, R, zeros(np, np), aa0, PP0, 1, 3, 0, ws1,[[1], [1], [1]])
=#

a0 = copy(a)
P0 = copy(P)
lik1 = KalmanFilterTools.kalman_likelihood(Y, Z, H, T, R, Q, a0, P0, 1, nobs, 0, ws)
@test a1 ≈ a0
@test P1 ≈ P0
@test lik0 ≈ ws.lik[1]  
