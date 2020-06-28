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

# alphah_t = a_t + Pstar_t*r0_{t-1} + Pinf_t*r1_{t-1}     (DK 5.24)
alphah = randn(ns)
a = randn(ns)
Pstar = randn(ns, ns)
Pinf = randn(ns, ns)
r0 = randn(ns)
r1 = randn(ns)
KalmanFilterTools.get_alphah!(alphah, a, Pstar, Pinf, r0, r1)
@test alphah ≈ a + Pstar*r0 + Pinf*r1

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

# D_t = KDKinf_t'*N0_t*KDKinf_t    (DK p. 135)
D = randn(ny, ny)
KDKinf = randn(ns, ny)
N0 = randn(ns, ns)
Tmp = randn(ny, ns)
KalmanFilterTools.get_D!(D, KDKinf,  N0, Tmp)
@test D ≈ KDKinf'*N0*KDKinf

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

# epsilon_t = -H_t*KDKinf*r0_t         (DK p. 135)
H = randn(ny, ny)
KDKinf = randn(ns, ny)
r0 = randn(ns)
tmp = randn(ny)
KalmanFilterTools.get_epsilonh!(epsilon, H, KDKinf, r0, tmp)
@test epsilon ≈ - H*KDKinf'*r0

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

# iFZ = inv(F)*Z
Z = randn(2, 5)
F = randn(2, 2)
F = F'*F
iFZ = randn(2, 5)
cholF = similar(F)
KalmanFilterTools.get_cholF!(cholF, F)
KalmanFilterTools.get_iFZ!(iFZ, cholF, Z)
@test iFZ ≈ F\Z

# iFZ = inv(F)*z
z = [2, 4]
F = randn(2, 2)
F = F'*F
iFZ = randn(2, 5)
cholF = similar(F)
KalmanFilterTools.get_cholF!(cholF, F)
KalmanFilterTools.get_iFZ!(iFZ, cholF, z)
Z = zeros(2, 5)
Z[1, 2] = 1.0
Z[2, 4] = 1.0
@test iFZ ≈ F\Z

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

# L_t = T - K(DK)_t*Z (DK 4.29)
z = [2, 3, 1]
Z = zeros(ny, ns)
for i=1:length(z)
    Z[i, z[i]] = 1.0
end
T = randn(ns, ns)
KDK = randn(ns, ny)
L = Matrix{Float64}(undef, ns, ns)
KalmanFilterTools.get_L!(L, T, KDK, Z)
@test L ≈ T - KDK*Z
KalmanFilterTools.get_L!(L, T, KDK, z)
@test L ≈ T - KDK*Z

Z = randn(ny, ns)
T = randn(ns, ns)
K = randn(ny, ns)
L = Matrix{Float64}(undef, ns, ns)
L1 = similar(L)
KalmanFilterTools.get_L_alternative!(L, T, K, Z, L1)
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

# Ptt = P - K'*Z*P
Ptt = randn(ns, ns)
P - randn(ns, ns)
K = randn(ny, ns)
ZP = randn(ny, ns)
KalmanFilterTools.get_updated_Ptt!(Ptt, P, K, ZP)
@test Ptt ≈ P - transpose(K)*ZP

# Pstartt = Pstar-Pstar*Z'*Kinf-Pinf*Z'*Kstar                           %(5.14) DK(2012)
Pstartt = randn(ns, ns)
Pstar = randn(ns, ns)
ZPstar = randn(ny, ns)
Kinf = randn(ny, ns)
ZPinf = randn(ny, ns)
Kstar = randn(ny, ns)
Pinftt = randn(ns, ns)
PTmp = randn(ns, ns)
KalmanFilterTools.get_updated_Pstartt!(Pstartt, Pstar, ZPstar, Kinf, ZPinf, Kstar, Pinftt, PTmp)
@test Pstartt ≈ Pstar - transpose(ZPstar)*Kinf - transpose(ZPinf)*Kstar

# V_t = P_t - P_t*N_{t-1}*P_t
V = Matrix{Float64}(undef, ns, ns)
#P = Matrix{Float64}(undef, ns, ns)
N1 = randn(ns, ns) #Matrix{Float64}(undef, ns, ns)
N1 = N1'N1
Ptmp = Matrix{Float64}(undef, ns, ns)
KalmanFilterTools.get_Valpha!(V, P, N1, Ptmp)
@test V ≈ P - P*N1*P

# Valpha_t = Pstar_t - Pstar_t*N0_{t-1}*Pstar_t
#            -(Pinf_t*N1_{t-1}*Pstar_t)'
#            -Pinf_t*N1_{t-1}*Pstar_t
#            -Pinf_t*N2_{t-1}*Pinf_t                       (DK 5.30)
Valpha = randn(ns, ns)
Pstar = randn(ns, ns)
Pstar = Pstar'*Pstar
Pinf = randn(ns, ns)
Pinf = Pinf'*Pinf
N0 = randn(ns, ns)
N0 = N0'*N0
N1 = randn(ns, ns)
N1 = N1'*N1
N2 = randn(ns, ns)
N2 = N2'*N2
Tmp = randn(ns, ns)
KalmanFilterTools.get_Valpha!(Valpha, Pstar, Pinf,N0, N1, N2, Tmp)
@test Valpha ≈ (Pstar - Pstar*N0*Pstar - (Pinf*N1*Pstar)'
                - Pinf*N1*Pstar - Pinf*N2*Pinf)

# QQ = R*Q*R'
R = randn(ns, np)
Q = randn(np, np)
Q = Q'*Q
QQ = zeros(ns, ns)
RQ = zeros(ns, np)
KalmanFilterTools.get_QQ!(QQ, R, Q, RQ)
@test QQ ≈ R*Q*R'

# att = a + K'*v
att = randn(ns)
a = randn(ns)
K = randn(ny, ns)
v = rand(ny)
KalmanFilterTools.get_updated_a!(att, a, K, v)
@test att ≈ a + transpose(K)*v

#v  = Y[:,t] - Z*a
a = randn(ns)
v = randn(ny)
y = randn(ny, 1)
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

# a = d + T*att
a = randn(ns)
d = randn(ns)
T = randn(ns, ns)
att = randn(ns)
KalmanFilterTools.update_a!(a, d, T, att)
@test a ≈ d + T*att

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
KalmanFilterTools.update_N!(N1, z, iFZ1, L, N, Ptmp)
ZiFZ = zeros(ns, ns)
ZiFZ[z, :] .= iFZ1
Z = zeros(ny, ns)
for i=1:length(z)
    Z[i, z[i]] = 1.0
end
@test ZiFZ ≈ Z'iFZ1
@test N1 ≈ ZiFZ + L'*N*L


# N0_{t-1} = L0_t'N0_t*L0_t (DK 5.29)
N0 = randn(ns, ns)
L0 = randn(ns, ns)
N0_1 = randn(ns, ns)
PTmp = randn(ns, ns)
KalmanFilterTools.update_N0!(N0, L0, N0_1, PTmp)
@test N0 ≈ L0'*N0_1*L0

# F1 = inv(Finf)
# N1_{t-1} = Z'*F1*Z + L0'*N1_t*L0 + L1'*N0_t*L0
N1 = randn(ns, ns)
Z = randn(ny, ns)
F = randn(ny, ny)
F = 0.5*(F + F')
iFZ = F\Z
L0 = randn(ns, ns)
N1_1 = randn(ns, ns)
L1 = randn(ns, ns)
N0_1 = randn(ns, ns)
PTmp = randn(ns, ns)
KalmanFilterTools.update_N1!(N1, Z, iFZ, L0, N1_1, L1, N0_1, PTmp)
@test iFZ ≈ inv(F)*Z
@test N1 ≈ Z'*inv(F)*Z + L0'*N1_1*L0 + L1'*N0_1*L0

# F2 = -inv(Finf)*Fstar*inv(Finv)
# N2_{t-1} = Z'*F2*Z + L0'*N2_t*L0 + L0'*N1_t*L1
#            + L1'*N1_t*L0 + L1'*N0_t*L1
N2 = randn(ns, ns)
Z = randn(ny, ns)
F = randn(ny, ny)
F = 0.5*(F + F')
iFZ = F\Z
Fstar = randn(ny, ny)
L0 = randn(ns, ns)
N0_1 = randn(ns, ns)
N1_1 = randn(ns, ns)
N2_1 = randn(ns, ns)
Tmp1 = randn(ny, ns)
Tmp2 = randn(ns, ns)
KalmanFilterTools.update_N2!(N2, iFZ, Fstar, L0, N2_1, N1_1,
                             L1, N0_1, Tmp1, Tmp2)
@test iFZ ≈ inv(F)*Z
@test N2 ≈ (-Z'*inv(F)*Fstar*inv(F)*Z + L0'*N2_1*L0 + L0'*N1_1*L1
            + L1'*N1_1*L0 + L1'*N0_1*L1)

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

# P = T*Ptt*T' + QQ
P = randn(ns, ns)
T = randn(ns, ns)
Ptt = randn(ns, ns)
QQ = randn(ns, ns)
PTmp = randn(ns, ns)
KalmanFilterTools.update_P!(P, T, Ptt, QQ, PTmp)
@test P ≈ T*Ptt*transpose(T) + QQ

# Pinf = T*Pinftt*T'
P = randn(ns, ns)
T = randn(ns, ns)
Ptt = randn(ns, ns)
PTmp = randn(ns, ns)
KalmanFilterTools.update_P!(P, T, Ptt, PTmp)
@test P ≈ T*Ptt*transpose(T)

# r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t
r = randn(ns)
r1 = randn(ns)
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

# rstar_{t-1} = Z_t'*iFinf_t*v_t + Linf_t'rstar_t + Lstar_t'*rinf_t      (DK 5.21)
rstar = randn(ns)
Z = randn(ny, ns)
iFv = randn(ny)
Linf = randn(ns, ns)
rstar1 = randn(ns)
Lstar = randn(ns, ns)
rinf1 = randn(ns)
KalmanFilterTools.update_r!(rstar, Z, iFv, Linf, rstar1, Lstar, rinf1)
@test rstar ≈ Z'*iFv + Linf'*rstar1 + Lstar'*rinf1

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

# Pstar  = T*(Pstar-Pstar*Z'*Kinf-Pinf*Z'*Kstar)*T'+QQ;         %(5.14) DK(2012)
Pstar = randn(ns, ns)
PstarOrig = copy(Pstar)
T = randn(ns, ns)
ZPinf = randn(ny, ns)
ZPstar = randn(ny, ns)
Kinf = randn(ny, ns)
Kstar = randn(ny, ns)
QQ = randn(ns, ns)
PTmp = randn(ns, ns)
KalmanFilterTools.update_Pstar!(Pstar, T, ZPinf, ZPstar, Kinf, Kstar, QQ, PTmp)
@test Pstar ≈ T*(PstarOrig-ZPstar'*Kinf-ZPinf'*Kstar)*T'+QQ

# Pinf   = T*(Pinf-Pinf*Z'*Kinf)*T';                             %(5.14) DK(2012)
Pinf = randn(ns, ns)
PinfOrig = copy(Pinf)
T = randn(ns, ns)
ZPinf = randn(ny, ns)
Kinf = randn(ny,ns)
PTmp = randn(ns, ns)
KalmanFilterTools.update_Pinf!(Pinf, T, ZPinf, Kinf, PTmp)
@test Pinf ≈ T*(PinfOrig - ZPinf'*Kinf)*T'

# rstar_{t-1} = Z_t'*iFinf_t*v_t + Linf_t'rstar_t + Lstar_t'*rinf_t (DK 5.21)
rstar = randn(ns)
Z = randn(ny, ns)
iFv = randn(ny)
Linf = randn(ns, ns)
rstar1 = randn(ns)
Lstar = randn(ns, ns)
rinf1 = randn(ns)
KalmanFilterTools.update_r!(rstar, Z, iFv, Linf, rstar1, Lstar, rinf1)
@test rstar ≈ Z'*iFv + Linf'*rstar1 + Lstar'*rinf1







    

