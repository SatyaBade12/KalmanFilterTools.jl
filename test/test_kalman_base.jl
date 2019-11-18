using LinearAlgebra
using LinearAlgebra.BLAS
using KalmanFilterTools
using Test

ny = 3
ns = 10
np   = 2
nobs = 50

Z = randn(ny, ns)
T = randn(ns, ns)
K = randn(ny, ns)
L = Matrix{Float64}(undef, ns, ns)
L1 = similar(L)

KalmanFilterTools.get_L!(L, T, K, Z, L1)
@test L ≈ T - T*K'*Z

z = [1, 3]
K1 = K[z, :]
K2 = zeros(ns, ns)
K2[z,:] .= K1
KalmanFilterTools.get_L!(L, T, K1, z, L1)
@test L ≈ T - T*K2'

# r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t
r = randn(ns)
r1 = similar(r)
iFv = randn(ny)

KalmanFilterTools.update_r!(r1, Z, iFv, L, r)
@test r1 ≈ Z'*iFv + L'r

# r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t
iFv1 = iFv[z]
KalmanFilterTools.update_r!(r1, z, iFv1, L, r)
ZiF = zeros(ns)
ZiF[z] .= iFv1
@test r1 ≈ ZiF + L'r

# alphah_t = a_t + P_t*r_{t-1}
alphah = Vector{Float64}(undef, ns)
a = randn(ns)
P = randn(ns, ns)
P = P'*P
KalmanFilterTools.get_alphah!(alphah, a, P, r1)
@test alphah ≈ a + P*r1

# N_{t-1} = Z_t'iF_t*Z_t + L_t'N_t*L_t
N = randn(ns, ns)
N = N'*N
N1 = similar(N)
iFZ = randn(ny, ns)
Ptmp = similar(P)
KalmanFilterTools.update_N!(N1, Z, iFZ, L, N, Ptmp)
@test N1 ≈ Z'*iFZ + L'*N*L

iFZ1 = iFZ[z, :]
ZiFZ = zeros(ns, ns)
ZiFZ[z, :] .= iFZ1
KalmanFilterTools.update_N!(N1, z, iFZ1, L, N, Ptmp)
@test N1 ≈ ZiFZ + L'*N*L

# V_t = P_t - P_t*N_{t-1}*P_t
V = Matrix{Float64}(undef, ns, ns)
KalmanFilterTools.get_V!(V, P, N1, Ptmp)
@test V ≈ P - P*N1*P

R = randn(ns, np)
Q = randn(np, np)
Q = Q'*Q
QQ = zeros(ns, ns)
RQ = zeros(ns, np)
KalmanFilterTools.get_QQ!(QQ, R, Q, RQ)
@test QQ ≈ R*Q*R'

H = randn(ny, ny)
H = H'*H
F = zeros(ny, ny)
ZP = zeros(ny, ns)
KalmanFilterTools.get_F!(F, ZP, Z, P, H)
@test ZP == Z*P
@test F ≈ Z*P*Z' + H

cholF = zeros(ny, ny)
KalmanFilterTools.get_cholF!(cholF, F)
CF = cholesky(0.5*(F + F'))
@test triu(cholF) ≈ CF.U

K = zeros(ny, ns)
KalmanFilterTools.get_K!(K, ZP, cholF)
@test K ≈ inv(F)*Z*P

P_0 = similar(P)
PTmp = similar(P)
copy!(P_0, P)
KalmanFilterTools.update_P!(P, T, QQ, K, ZP, PTmp)
@test P ≈ T*(P_0 - K'*ZP)*T' + QQ

W = rand(ns, ny)
K = Z*P
mul!(W, T, transpose(K))
@test W ≈ T*P*Z'

M = rand(ny, ny)
ZW = rand(ny, ny)
KalmanFilterTools.get_M!(M, cholF, ZW) 
@test M ≈ -inv(F)

#v  = Y[:,t] - Z*a
a = rand(ns)
v = rand(ny)
y = rand(ny, 1)
KalmanFilterTools.get_v!(v, y, Z, a, 1, ny)
@test v ≈ y[:,1] - Z*a

# iFv = inv(F)*v
iFv = similar(v)
KalmanFilterTools.get_iFv!(iFv, cholF, v)
@test iFv ≈ F\v

# a = T(a + K'*iFv)
a_0 = copy(a)
a1 = similar(a)
KalmanFilterTools.update_a!(a, K, iFv, a1, T)
@test a ≈ T*(a_0 + K'*iFv)

# M = M + M*W'*Z'iF*Z*W*M
M_0 = copy(M)
ZWM = similar(M)
iFZWM = similar(M)
KalmanFilterTools.update_M!(M, Z, W, cholF, ZW, ZWM, iFZWM)
@test M ≈ M_0 + M_0*W'*Z'*inv(F)*Z*W*M_0

# F =  F + Z*W*M*W'Z'
F_0 = copy(F)
gemm!('N', 'T', 1.0, ZWM, ZW, 1.0,F)
@test F ≈ F_0 + Z*W*M_0*W'*Z'

# K = K + Z*W*M*W'
K_0 = copy(K)
KalmanFilterTools.update_K!(K, ZWM, W)
@test K ≈ K_0 + Z*W*M_0*W'

# W = T(W - K'*iF*Z*W)
#K2 = K1  + T*W*M_0*W'*Z'
W_0 = copy(W)
ZW = Z*W
F = randn(ny, ny)
F = F'*F
cholF = copy(F)
LAPACK.potrf!('U', cholF)
iFZW = rand(ny, ny)
copy!(iFZW, ZW)
LAPACK.potrs!('U', cholF, iFZW)
@test iFZW ≈ inv(F)*ZW

KtiFZW = rand(ns, ny)
KalmanFilterTools.update_W!(W, ZW, cholF, T, K, iFZW, KtiFZW)
@test iFZW ≈ inv(F)*ZW
@test KtiFZW ≈ W_0 - K'*inv(F)*Z*W_0
@test W ≈ T*(W_0 - K'*inv(F)*Z*W_0)
#@test ws1.W ≈ T*Wold - K2*inv(ws1.F)*Z*Wold

# Z as selection matrix
fill!(Z, 0.0)
Z[1, 4] = 1
Z[2, 3] = 1
Z[3, 2] = 1
z = [4, 3, 2]

P = copy(P_0)
KalmanFilterTools.get_F!(F, ZP, z, P, H)
@test F ≈ Z*P*Z' + H

# v  = Y[:,t] - Z*a
a = copy(a_0)
KalmanFilterTools.get_v!(v, y, z, a, 1, ny)
@test v ≈ y[:,1] - Z*a_0

# missing observations
vv = view(v, 1:ny)
c = randn(ny)
vc = view(c, 1:ny)
vZ = view(Z, 1:ny, :)
va = view(a_0, :)
full_data_pattern .= [collect(1:ny)]
pattern = full_data_pattern[1]

KalmanFilterTools.get_v!(vv, y, vc, vZ, va, 1, pattern)
@test vv ≈ y[:, 1] - c - Z*a

va1 = zeros(ns)
va = randn(ns)
vd = zeros(ns)
vK = randn(ny,ns)
vv = randn(ny)
vT = view(T, :, :)
KalmanFilterTools.update_a!(va1, va, vd, vK, vv, a1, vT)
@test va1  ≈ vd + vT*(va + vK'*vv)

# P = T*(P - K'*Z*P)*T'+ QQ
vP = copy(P_0)
vZP = randn(ny, ns)
vP1 = copy(vP)
Ptmp = similar(P)
KalmanFilterTools.update_P!(vP1, vT, QQ, vK, vZP, PTmp)
@test vP1 ≈ vT*(vP - vK'*vZP)*vT' + QQ

AA = randn(ns, ns)
Pinf = AA*diagm(rand(ns))*AA'
BB = randn(ns, ns)
Pstar = BB*diagm(rand(ns))*BB'

F = zeros(ny, ny)
ZP = zeros(ny, ns)
KalmanFilterTools.get_F!(F, ZP, Z, Pinf)
@test F ≈ Z*Pinf*Z'
@test ZP ≈ Z*Pinf

cholF = 0.5*(F + F')
info = LAPACK.potrf!('U', cholF)
@test info[2] == 0

K = F\Z*Pinf
Kstar = similar(K)
Fstar = zeros(ny, ny)
ZPstar = zeros(ny, ns)
KalmanFilterTools.get_F!(Fstar, ZPstar, Z, Pstar, H)
@test Fstar ≈ Z*Pstar*Z' + H

KalmanFilterTools.get_Kstar!(Kstar, Z, Pstar, Fstar, K, cholF)
@test Kstar ≈ F\(Z*Pstar - Fstar*K)

z = [4, 3, 2]
F = 0.5*(F + F')
cholF = copy(F)
LAPACK.potrf!('U', cholF)
KalmanFilterTools.get_Kstar!(Kstar, z, Pstar, Fstar, K, cholF)
@test Kstar ≈ F\(Z*Pstar - Fstar*K)

QQ = rand(ns, ns)
RQ = rand(ns, np)
KalmanFilterTools.get_QQ!(QQ, R, Q, RQ)
@test QQ ≈ R*Q*R'

