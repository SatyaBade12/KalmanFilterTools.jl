using LinearAlgebra
using LinearAlgebra.BLAS
using KalmanFilterTools
using MAT
using Test

path = dirname(@__FILE__)

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
#@test a1 ≈ a0
#@test P1 ≈ P0
#@test lik0 ≈ ws.lik[1]  

vars = matread("$path/reference/test_data.mat")

Y = vars["Y"]
Z = vars["Z"]
H = vars["H"]
T = vars["T"]
R = vars["R"]
Q = vars["Q"]
Pinf_0 = vars["Pinf"]
Pstar_0 = vars["Pstar"]

ny, nobs = size(Y)
ns, np = size(R)

a_0 = zeros(ns)
if H == 0
    H = zeros(ny, ny)
end

full_data_pattern = [collect(1:ny) for o = 1:nobs]

aa = zeros(ns, nobs + 1)
aa[:, 1] .= a_0
att = similar(aa)
Pinf = zeros(ns, ns, nobs + 1)
Pinftt = zeros(ns, ns, nobs + 1)
Pstar = zeros(ns, ns, nobs + 1)
Pstartt = zeros(ns, ns, nobs + 1)
Pinf[:, :, 1] = Pinf_0
Pinftt[:, :, 1] = Pinf_0
Pstar[:, :, 1] = Pstar_0
Pstartt[:, :, 1] =  Pstar_0
alphah = zeros(ns, nobs)
epsilonh = zeros(ny, nobs)
etah = zeros(np, nobs)
Valphah = zeros(ns, ns, nobs)
Vepsilonh = zeros(ny, ny, nobs)
Vetah = zeros(np, np, nobs)
c = zeros(ny)
d = zeros(ns)

ws6 = DiffuseKalmanSmootherWs(ny, ns, np, nobs)
llk_6b = diffuse_kalman_smoother!(Y, c, Z, H, d, T, R, Q, aa, att,
                                  Pinf, Pinftt, Pstar, Pstartt,
                                  alphah, epsilonh, etah, Valphah,
                                  Vepsilonh, Vetah, 1, nobs, 0,
                                  1e-8, ws6)


aa = zeros(ns, nobs + 1)
aa[:, 1] .= a_0
att = similar(aa)
Pinf = zeros(ns, ns, nobs + 1)
Pinftt = zeros(ns, ns, nobs + 1)
Pstar = zeros(ns, ns, nobs + 1)
Pstartt = zeros(ns, ns, nobs + 1)
Pinf[:, :, 1] = Pinf_0
Pinftt[:, :, 1] = Pinf_0
Pstar[:, :, 1] = Pstar_0
Pstartt[:, :, 1] =  Pstar_0
alphah = zeros(ns, nobs)
epsilonh = zeros(ny, nobs)
etah = zeros(np, nobs)
Valphah = zeros(ns, ns, nobs)
Vepsilonh = zeros(ny, ny, nobs)
Vetah = zeros(np, np, nobs)
c = zeros(ny)
d = zeros(ns)
r0 = randn(ns)
r0_1 = randn(ns)
r1 = randn(ns)
r1_1 = randn(ns)
L0 = Matrix{Float64}(undef, ns, ns)
L1 = similar(L0)
N0 = similar(L0)
N0_1 = similar(L0)
N1 = similar(L0)
N1_1 = similar(L0)
N2 = similar(L0)
N2_1 = similar(L0)
v = randn(ny)
ws = DiffuseKalmanSmootherWs(ny, ns, np, nobs)

y = Y[:, 1]
t = 1
a0 = copy(aa[:, 1])
a = copy(a0)
pinf0 = copy(Pinf[:, :, 1])
pinf = copy(pinf0)
pstar0 = copy(Pstar[:, :, 1])
pstar = copy(pstar0)
QQ = R*Q*R'
KalmanFilterTools.univariate_step(y, t, c, Z, H, d, T, QQ, a, pinf, pstar, 1e-10, 1e-10, ws)
v = y - c - Z*a0
K0 = T*pinf0*Z'*inv(Z*pinf0*Z')
a1 = T*a0 + K0*v
@test a ≈ a1

Finf0 = Z*pinf0*Z'
Finf = copy(Finf0)
Fstar0 = Z*pstar0*Z' + H
Fstar = copy(Fstar0)

Kinf0 = pinf0*Z'*inv(Finf0)
Kinf = copy(Kinf0)
K0 = pstar0*Z'*inv(Finf0) - pinf0*Z'*inv(Finf0)*Fstar0*inv(Finf0) 
K = copy(K0)

L0_target = I(ns) - Kinf0*Z
L1_target = -K0*Z
r1_target = Z'inv(Finf)*v + L0_target'*r1 +L1_target'*r0
r0_target = L0_target'*r0
N0_target = L0_target'*N0*L0_target
N1_target = Z'*inv(Finf)*Z + L0_target'*N1*L0_target + L1_target'*N0_target*L0_target
F2 = -inv(Finf)*Fstar*inv(Finf)
N2_target = Z'F2*Z + L0_target'*N2*L0_target' + L0_target'*N1_target*L1_target + L1_target*N1_target*L0_target + L1_target'*N0_target*L1_target 

tol = 1e-12
println("Smoothing")
KalmanFilterTools.univariate_diffuse_smoother_step!(T, ws.F[:, :, 1], ws.Fstar[:, :, 1],
                                                    ws.Kinf[:, :, 1], ws.K[:, :, 1],
                                                    ws.L, ws.L1, ws.N, ws.N1,
                                                    ws.N2, r0, r1, ws.v[:,1], Z,
                                                    tol, ws)

@test Z'inv(Finf)*ws.v[:,1] ≈ (I(ns) - ws.Kinf[1, :, 1]*transpose(Z[1, :])/ws.F[1,1,1])*Z[2,:]*ws.v[2,1]/ws.F[2,2,1] + Z[1,:]*ws.v[1,1]/ws.F[1,1,1]



#@test L0 ≈ L0_target
#@test L1 ≈ L1_target
#@test N0 ≈ N0_target
#@test N1 ≈ N1_target
#@test N2 ≈ N2_target
@test r0 ≈ transpose(T)*r0_target
@test r1 ≈ transpose(T)*r1_target

