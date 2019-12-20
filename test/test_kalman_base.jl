using LinearAlgebra
using LinearAlgebra.BLAS
using KalmanFilterTools
using Test

ny = 3
ns = 10
np   = 2
nobs = 50

a = randn(nobs, nobs)
achol = cholesky(a'*a)
d = KalmanFilterTools.det_from_cholesky(achol.U)
@test det(a'*a) ≈ d 

# alphah_t = a_t + P_t*r_{t-1}
alphah = Vector{Float64}(undef, ns)
a = randn(ns)
P = randn(ns, ns)
P = P'*P
r1 = randn(ns)
KalmanFilterTools.get_alphah!(alphah, a, P, r1)
@test alphah ≈ a + P*r1

F = randn(ny, ny)
F = F'*F
cholF = zeros(ny, ny)
KalmanFilterTools.get_cholF!(cholF, F)
CF = cholesky(0.5*(F + F'))
@test triu(cholF) ≈ CF.U

# D = inv(F_t) + K_t*T*N_t*T'*K'
D = randn(ny, ny)
F = randn(ny, ny)
F = F'*F
cholF = cholesky(F)
K = randn(ny, ns)
T = randn(ns, ns)
N = randn(ns, ns)
KT = randn(ny, ns)
tmp = randn(ny, ns)
KalmanFilterTools.get_D!(D, cholF.U, K, T, N, KT, tmp)
@test D ≈ inv(F) + K*T*N*T'*K'

# epsilonh_t = H*(iF_t*v_t - K_t*T*r_t)
epsilon = randn(ny)
H = randn(ny, ny)
v = randn(ny)
iFv = randn(ny)
K = randn(ny, ns)
T = randn(ns, ns)
r = randn(ns)
tmp1 = zeros(ny)
tmp2 = zeros(ns)
KalmanFilterTools.get_epsilonh!(epsilon, H, iFv, K, T, r, tmp1, tmp2)
@test epsilon ≈ H*(iFv - K*T*r)

# etah = Q*R'*r_t
eta = randn(np)
Q = randn(np, np)
R = randn(ns, np)
r = randn(ns)
tmp = zeros(np)
KalmanFilterTools.get_etah!(eta, Q, R, r, tmp)
@test eta ≈ Q*R'*r

H = randn(ny, ny)
H = H'*H
F = zeros(ny, ny)
ZP = zeros(ny, ns)
Z = randn(ny, ns)
KalmanFilterTools.get_F!(F, ZP, Z, P, H)
@test ZP == Z*P
@test F ≈ Z*P*Z' + H

# Z as selection matrix
fill!(Z, 0.0)
Z[1, 4] = 1
Z[2, 3] = 1
Z[3, 2] = 1
z = [4, 3, 2]
P_0 = randn(ns, ns)
P = copy(P_0)
KalmanFilterTools.get_F!(F, ZP, z, P, H)
@test F ≈ Z*P*Z' + H

F = zeros(ny, ny)
ZP = zeros(ny, ns)
Pinf = randn(ns, ns)
KalmanFilterTools.get_F!(F, ZP, Z, Pinf)
@test F ≈ Z*Pinf*Z'
@test ZP ≈ Z*Pinf

# iFv = inv(F)*v
F = randn(ny, ny)
F = F'*F
cholF = similar(F)
KalmanFilterTools.get_cholF!(cholF, F)
v = randn(ny)
iFv = similar(v)
KalmanFilterTools.get_iFv!(iFv, cholF, v)
@test iFv ≈ F\v

K = zeros(ny, ns)
Z = randn(ny, ns)
P = randn(ns, ns)
ZP = Z*P
F = randn(ny, ny)
F = F'*F
cholF = similar(F)
KalmanFilterTools.get_cholF!(cholF, F)
KalmanFilterTools.get_K!(K, ZP, cholF)
@test K ≈ inv(F)*Z*P

K = F\Z*Pinf
Kstar = similar(K)
Fstar = zeros(ny, ny)
ZPstar = zeros(ny, ns)
Z = randn(ny, ns)
Pstar = randn(ns, ns)
H = randn(ny, ny)
KalmanFilterTools.get_F!(Fstar, ZPstar, Z, Pstar, H)
@test Fstar ≈ Z*Pstar*Z' + H

KalmanFilterTools.get_Kstar!(Kstar, Z, Pstar, Fstar, K, cholF)
@test Kstar ≈ F\(Z*Pstar - Fstar*K)

Kstar = randn(ny, ns)
z = [4, 3, 2]
F = 0.5*(F + F')
cholF = copy(F)
LAPACK.potrf!('U', cholF)
KalmanFilterTools.get_Kstar!(Kstar, z, Pstar, Fstar, K, cholF)
@test Kstar ≈ F\(Pstar[z,:] - Fstar*K)

Z = randn(ny, ns)
T = randn(ns, ns)
K = randn(ny, ns)
L = Matrix{Float64}(undef, ns, ns)
L1 = similar(L)
KalmanFilterTools.get_L!(L, T, K, Z, L1)
@test L ≈ T - T*K'*Z

z = [1, 3, 2]
K1 = K[z, :]
K2 = zeros(ns, ns)
K2[z,:] .= K1
KalmanFilterTools.get_L!(L, T, K1, z, L1)
@test L ≈ T - T*K2'

M = rand(ny, ny)
ZW = rand(ny, ny)
KalmanFilterTools.get_M!(M, cholF, ZW) 
@test M ≈ -inv(F)

# V_t = P_t - P_t*N_{t-1}*P_t
V = Matrix{Float64}(undef, ns, ns)
P = Matrix{Float64}(undef, ns, ns)
N1 = Matrix{Float64}(undef, ns, ns)
Ptmp = Matrix{Float64}(undef, ns, ns)
KalmanFilterTools.get_Valpha!(V, P, N1, Ptmp)
@test V ≈ P - P*N1*P

R = randn(ns, np)
Q = randn(np, np)
Q = Q'*Q
QQ = zeros(ns, ns)
RQ = zeros(ns, np)
KalmanFilterTools.get_QQ!(QQ, R, Q, RQ)
@test QQ ≈ R*Q*R'

#v  = Y[:,t] - Z*a
a = rand(ns)
v = rand(ny)
y = rand(ny, 1)
KalmanFilterTools.get_v!(v, y, Z, a, 1, ny)
@test v ≈ y[:,1] - Z*a

# missing observations
vv = view(v, 1:ny)
c = randn(ny)
vc = view(c, 1:ny)
vZ = view(Z, 1:ny, :)
a_0 = randn(ns)
va = view(a_0, :)
full_data_pattern = [collect(1:ny)]
pattern = full_data_pattern[1]
KalmanFilterTools.get_v!(vv, y, vc, vZ, va, 1, pattern)
@test vv ≈ y[:, 1] - c - Z*a_0

# Vepsilon_t = H - H*D_t*H
Vepsilon = zeros(ny, ny)
H = randn(ny, ny)
D = randn(ny, ny)
tmp = zeros(ny, ny)
KalmanFilterTools.get_Vepsilon!(Vepsilon, H, D, tmp)
@test Vepsilon ≈ H - H*D*H 

# Veta_t = Q - Q*R'*N_t*R*Q
Veta = zeros(np, np)
Q = zeros(np, np)
R = randn(ns, np)
N = randn(ns, ns)
RQ = zeros(ns, np)
tmp = zeros(ns, np)
KalmanFilterTools.get_Veta!(Veta, Q, R, N, RQ, tmp)
@test Veta ≈ Q - Q*R'*N*R*Q


Zsmall = randn(ny, ns)
iZsmall = rand(Int, ny)
Z = randn(ny, ns)
pattern = [1, 3]
n = length(pattern)
vZsmall = KalmanFilterTools.get_vZsmall(Zsmall, iZsmall, Z, pattern, n, ny)
@test all(vZsmall .== Z[pattern, :])

Zsmall = randn(ny, ns)
iZsmall = rand(Int, ny)
Z = [4, 3, 6]
pattern = [1, 3]
n = length(pattern)
vZsmall = KalmanFilterTools.get_vZsmall(Zsmall, iZsmall, Z, pattern, n, ny)
@test all(vZsmall .== Z[pattern])

# a = T(a + K'*iFv)
a_0 = copy(a)
a1 = similar(a)
KalmanFilterTools.update_a!(a, K, iFv, a1, T)
@test a ≈ T*(a_0 + K'*iFv)

va1 = zeros(ns)
va = randn(ns)
vd = zeros(ns)
vK = randn(ny,ns)
vv = randn(ny)
vT = view(T, :, :)
KalmanFilterTools.update_a!(va1, va, vd, vK, vv, a1, vT)
@test va1  ≈ vd + vT*(va + vK'*vv)

# K = K + Z*W*M*W'
K = randn(ny, ns)
K_0 = copy(K)
ZWM = randn(ny, ny)
W = randn(ns, ny)
KalmanFilterTools.update_K!(K, ZWM, W)
@test K ≈ K_0 + ZWM*W'

# M = M + M*W'*Z'iF*Z*W*M
M_0 = copy(M)
Z = randn(ny, ns)
ZWM = similar(M)
iFZWM = similar(M)
KalmanFilterTools.update_M!(M, Z, W, cholF, ZW, ZWM, iFZWM)
@test M ≈ M_0 + M_0*W'*Z'*inv(F)*Z*W*M_0

# N_{t-1} = Z_t'iF_t*Z_t + L_t'N_t*L_t
Z = randn(ny, ns)
N = randn(ns, ns)
N = N'*N
N1 = similar(N)
iFZ = randn(ny, ns)
L = randn(ns, ns)
Ptmp = similar(P)
KalmanFilterTools.update_N!(N1, Z, iFZ, L, N, Ptmp)
@test N1 ≈ Z'*iFZ + L'*N*L

iFZ1 = iFZ[z, :]
ZiFZ = zeros(ns, ns)
ZiFZ[z, :] .= iFZ1
KalmanFilterTools.update_N!(N1, z, iFZ1, L, N, Ptmp)
@test N1 ≈ ZiFZ + L'*N*L


P_0 = similar(P)
PTmp = similar(P)
copy!(P_0, P)
KalmanFilterTools.update_P!(P, T, QQ, K, ZP, PTmp)
@test P ≈ T*(P_0 - K'*ZP)*T' + QQ

# P = T*(P - K'*Z*P)*T'+ QQ
vP = copy(P_0)
vZP = randn(ny, ns)
vP1 = copy(vP)
Ptmp = similar(P)
KalmanFilterTools.update_P!(vP1, vT, QQ, vK, vZP, PTmp)
@test vP1 ≈ vT*(vP - vK'*vZP)*vT' + QQ

# r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t
r = randn(ns)
r1 = similar(r)
Z = randn(ny, ns)
iFv = randn(ny)
KalmanFilterTools.update_r!(r1, Z, iFv, L, r)
@test r1 ≈ Z'*iFv + L'r

# r_{t-1} = z_t'*iF_t*v_t + L_t'r_t
z = [3, 2]
iFv1 = iFv[z]
KalmanFilterTools.update_r!(r1, z, iFv1, L, r)
ZiF = zeros(ns)
ZiF[z] .= iFv1
@test r1 ≈ ZiF + L'r

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

# F =  F + Z*W*M*W'Z'
F = randn(ny, ny)
F_0 = copy(F)
ZWM = randn(ny, ny)
ZW = randn(ny, ny)
gemm!('N', 'T', 1.0, ZWM, ZW, 1.0,F)
@test F ≈ F_0 + ZWM*ZW'



