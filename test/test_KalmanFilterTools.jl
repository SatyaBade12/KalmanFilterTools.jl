using HDF5
using LinearAlgebra
using LinearAlgebra.BLAS
using KalmanFilterTools
using MAT
using Test

path = dirname(@__FILE__)

# Using test case from NYFED package StateSpaceRoutines
# Initialize arguments to function
file = h5open("$path/reference/kalman_filter_args.h5", "r")
y = read(file, "data")
T, R, C    = read(file, "TTT"), read(file, "RRR"), read(file, "CCC")
Q, Z, D, H = read(file, "QQ"), read(file, "ZZ"), read(file, "DD"), read(file, "EE")
s_0, P_0   = read(file, "z0"), read(file, "P0")
close(file)

ny, ns = size(Z)
nobs = size(y, 2)
np = size(R, 2)

# Removing measure equation constant from observations
y .-= D

# Create data_pattern for all observations are available

full_data_pattern = [collect(1:ny) for o = 1:nobs]

# Simple Kalman Filter
P = copy(P_0)
s = copy(s_0)
@testset "Basic Kalman Filter" begin
    ws1 = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)

    copy!(P, P_0)
    llk_1 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)

    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ llk_1
    end
    
    copy!(s, s_0)
    copy!(P, P_0)
    llk_2 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_2 ≈ llk_1

    copy!(s, s_0)
    copy!(P, P_0)
    llk_2 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_2 ≈ llk_1

    copy!(s, s_0)
    copy!(P, P_0)
    llk_2 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_2 ≈ llk_1

end

H_0 = copy(H)
# Singular F matrix
@testset "Singular F matrix diagonal H" begin
    H = zeros(ny, ny) + I(ny)
    
    ws1 = KalmanLikelihoodWs(ny, ns, np, nobs)
    P = zeros(ns, ns, nobs+1)
    s = copy(s_0)

    llk_1 = kalman_filter!(y, zeros(ny), Z, H, zeros(ns), T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test P[:, :, 2] ≈ R*Q*R'

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_2 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_2  ≈ llk_1

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_3 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_3  ≈ llk_1

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_4 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_4  ≈ llk_1

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_5 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_5  ≈ llk_1

end

@testset "Singular F matrix Full H" begin
    H = randn(ny, ny)
    H = H'*H
    
    ws1 = KalmanLikelihoodWs(ny, ns, np, nobs)
    P = zeros(ns, ns, nobs+1)
    s = copy(s_0)

    llk_1 = kalman_filter!(y, zeros(ny), Z, H, zeros(ns), T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test P[:, :, 2] ≈ R*Q*R'

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_2 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_2  ≈ llk_1

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_3 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_3  ≈ llk_1

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_4 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_4  ≈ llk_1

    P = zeros(ns, ns)
    s = copy(s_0)
    
    llk_5 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_5  ≈ llk_1

end

H = copy(H_0)    
# Fast Kalman Filter
@testset "Fast Kalman Filter" begin
    ws1 = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)
    ws2 = FastKalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)
    P = copy(P_0)
    s = copy(s_0)
    

    copy!(s, s_0)
    llk_1 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)

    copy!(P, P_0)
    copy!(s, s_0)
    llk_2 = fast_kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws2)
    @test llk_2 ≈ llk_1

    copy!(P, P_0)
    copy!(s, s_0)
    llk_3 = fast_kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws2, full_data_pattern)
    @test llk_3 ≈ llk_1

end


@testset "Z as selection matrix" begin
    ws1 = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)
    ws2 = FastKalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)

    fill!(Z, 0.0)
    Z[1, 4] = 1
    Z[2, 3] = 1
    Z[3, 2] = 1
    z = [4, 3, 2]

    s = copy(s_0)
    P = copy(P_0)
    llk_1 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    
    s = copy(s_0)
    P = copy(P_0)
    llk_2 = kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_1 ≈ llk_2

    s = copy(s_0)
    P = copy(P_0)
    llk_2 = kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_1 ≈ llk_2

    s = copy(s_0)
    P = copy(P_0)
    llk_3 = fast_kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws2)
    @test llk_1 ≈ llk_3

    s = copy(s_0)
    P = copy(P_0)
    llk_3 = fast_kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws2, full_data_pattern)
    @test llk_1 ≈ llk_3

end

@testset "Kalman Filter" begin
    c = zeros(ny)
    d = zeros(ns)
    s = copy(s_0)
    P = copy(P_0)
    nobs1 = 1
    ws1 = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs1)

    kalman_filter!(y, c, Z, H, d, T, R, Q, s, P, 1, nobs1, 0, ws1, full_data_pattern)
    
    cs = zeros(ny, nobs)
    Zs = zeros(ny, ns, nobs)
    Hs = zeros(ny, ny, nobs)
    ds = zeros(ns, nobs)
    Ts = zeros(ns, ns, nobs)
    Rs = zeros(ns, np, nobs)
    Qs = zeros(np, np, nobs)
    ss = zeros(ns, nobs+1)
    Ps = zeros(ns, ns, nobs)

    for i = 1:nobs
        cs[:, i] = c
        Zs[:, :, i] = Z
        Hs[:, :, i] = H
        ds[:, i] = d
        Ts[:, :, i] = T
        Rs[:, :, i] = R
        Qs[:, :, i] = Q
    end

                    
    ss[:, 1] = s_0
    Ps[:, :, 1] = P_0
    kalman_filter!(y, cs, Zs, Hs, ds, Ts, Rs, Qs, ss, Ps, 1, nobs1, 0, ws1, full_data_pattern)
    @test ss[:, nobs1+1] ≈ s
    @test Ps[:, : , nobs1+1] ≈ P

end

# Replication data computed with Dynare
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

@testset "Diffuse Kalman Filter" begin
    ws4 = DiffuseKalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)
    
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    copy!(ws4.QQ, R*Q*R')
    
    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, Z, H, T, ws4.QQ, a, Pinf, Pstar, 1, nobs, 1e-8, ws4)
    llk_3 = -0.5*(t*ny*log(2*pi) + sum(ws4.lik[1:t]))

    # Dynare returns minus log likelihood
    @test llk_3 ≈ -vars["dLIK"]
    @test a ≈ vars["a"]
    @test Pstar ≈ vars["Pstar1"]

    z = [4, 3]
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, z, H, T, ws4.QQ, a, Pinf, Pstar, 1, nobs, 1e-8, ws4)
    llk_3 = -0.5*(t*ny*log(2*pi) + sum(ws4.lik[1:t]))

    # Dynare returns minus log likelihood
    @test llk_3 ≈ -vars["dLIK"]
    @test a ≈ vars["a"]
    @test Pstar ≈ vars["Pstar1"]

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_4 = diffuse_kalman_likelihood(Y, Z, H, T, R, Q, a, Pinf, Pstar, 1, nobs, 0, 1e-8, ws4)

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_5 = diffuse_kalman_likelihood(Y, z, H, T, R, Q, a, Pinf, Pstar, 1, nobs, 0, 1e-8, ws4)
    @test llk_5 ≈ llk_4 

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)

    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, Z, H, T, ws4.QQ, a, Pinf, Pstar, 1, nobs, 1e-8, ws4, full_data_pattern)
    llk_3 = -0.5*(t*ny*log(2*pi) + sum(ws4.lik[1:t]))

    # Dynare returns minus log likelihood
    @test llk_3 ≈ -vars["dLIK"]
    @test a ≈ vars["a"]
    @test Pstar ≈ vars["Pstar1"]

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, z, H, T, ws4.QQ, a, Pinf, Pstar, 1, nobs, 1e-8, ws4, full_data_pattern)
    llk_3 = -0.5*(t*ny*log(2*pi) + sum(ws4.lik[1:t]))

    # Dynare returns minus log likelihood
    @test llk_3 ≈ -vars["dLIK"]
    @test a ≈ vars["a"]
    @test Pstar ≈ vars["Pstar1"]

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_4 = diffuse_kalman_likelihood(Y, Z, H, T, R, Q, a, Pinf, Pstar, 1, nobs, 0, 1e-8, ws4, full_data_pattern)

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_5 = diffuse_kalman_likelihood(Y, z, H, T, R, Q, a, Pinf, Pstar, 1, nobs, 0, 1e-8, ws4, full_data_pattern)
    @test llk_5 ≈ llk_4 
end

@testset "start and last" begin
    nobs1 = nobs - 1
    ws1 = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)

    P_0 = randn(ns, ns)
    P_0 = P_0'P_0
    
    a = copy(a_0)
    P = copy(P_0)
    llk_1 = kalman_likelihood(Y[:,2:nobs1], Z, H, T, R, Q, a, P, 1, nobs-2, 0, ws1)
    copy!(a, a_0)
    copy!(P, P_0)
    llk_2 = kalman_likelihood(Y, Z, H, T, R, Q, a, P, 2, nobs1, 0, ws1)
    @test llk_2 ≈ llk_1

    copy!(a, a_0)
    copy!(P, P_0)
    llk_1 = kalman_likelihood_monitored(Y[:,2:nobs1], Z, H, T, R, Q, a, P, 1, nobs-2, 0, ws1)
    copy!(a, a_0)
    copy!(P, P_0)
    llk_2 = kalman_likelihood_monitored(Y, Z, H, T, R, Q, a, P, 2, nobs1, 0, ws1)
    @test llk_2 ≈ llk_1

    copy!(a, a_0)
    copy!(P, P_0)
    llk_1 = kalman_likelihood(Y[:,2:nobs1], Z, H, T, R, Q, a, P, 1, nobs-2, 0, ws1, full_data_pattern)
    copy!(a, a_0)
    copy!(P, P_0)
    llk_2 = kalman_likelihood(Y, Z, H, T, R, Q, a, P, 2, nobs-1, 0, ws1, full_data_pattern)
    @test llk_2 ≈ llk_1

    copy!(a, a_0)
    copy!(P, P_0)
    llk_1 = kalman_likelihood_monitored(Y[:,2:nobs1], Z, H, T, R, Q, a, P, 1, nobs-2, 0 , ws1, full_data_pattern)
    copy!(a, a_0)
    copy!(P, P_0)
    llk_2 = kalman_likelihood_monitored(Y, Z, H, T, R, Q, a, P, 2, nobs-1, 0, ws1, full_data_pattern)
    @test llk_2 ≈ llk_1

    ws4 = DiffuseKalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)
    
    z = [4, 3]
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    copy!(ws4.QQ, R*Q*R')

    llk_4 = diffuse_kalman_likelihood(Y[:,2:nobs1], z, H, T, R, Q, a, Pinf, Pstar, 1, nobs-2, 0, 1e-8, ws4, full_data_pattern)

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_5 = diffuse_kalman_likelihood(Y, z, H, T, R, Q, a, Pinf, Pstar, 2, nobs-1, 0, 1e-8, ws4, full_data_pattern)
    @test llk_5 ≈ llk_4 
end
nothing

