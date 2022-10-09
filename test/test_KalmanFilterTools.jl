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
Q, Z, D, H_0 = read(file, "QQ"), read(file, "ZZ"), read(file, "DD"), read(file, "EE")
s_0, P_0   = read(file, "z0"), read(file, "P0")
close(file)

function t_init!(H, P, s, H_0, P_0, s_0)
    copy!(H, H_0)
    copyto!(P, 1, P_0, 1, ns*ns)
    copyto!(s, 1, s_0, 1, ns)
end
    
ny, ns = size(Z)
nobs = size(y, 2)
np = size(R, 2)

# Removing measure equation constant from observations
y .-= D

# Create data_pattern for all observations are available

full_data_pattern = [collect(1:ny) for o = 1:nobs]

H = zeros(ny, ny)
P = zeros(ns, ns)
s = zeros(ns)
# Simple Kalman Filter
@testset "Basic Kalman Filter" begin
    ws1 = KalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_1 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)

    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ llk_1
    end
    
    t_init!(H, P, s, H_0, P_0, s_0)
    llk_2 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_2 ≈ llk_1

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_2 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_2 ≈ llk_1

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_2 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_2 ≈ llk_1

end

@testset "Singular F matrix" begin
    H_1 = zeros(ny, ny)
    
    ws1 = KalmanLikelihoodWs(ny, ns, np, nobs)
    ws2 = KalmanFilterWs(ny, ns, np, nobs)
    P = zeros(ns, ns, nobs + 1)
    s = zeros(ns, nobs + 1)
    s[:, 1] .= s_0
    Ptt = zeros(ns, ns, nobs)
    stt = zeros(ns, nobs)

    llk_1 = kalman_filter!(y, zeros(ny), Z, H_1, zeros(ns), T, R, Q, s, stt, P, Ptt, 1, nobs, 0, ws2, full_data_pattern)
    @test P[:, :, 2] ≈ R*Q*R'

    P = zeros(ns, ns)
    P_0 = zeros(ns, ns)
    s = zeros(ns)
    t_init!(H, P, s, H_1, P_0, s_0)

    llk_2 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_2  ≈ llk_1

    t_init!(H, P, s, H_1, P_0, s_0)
    
    llk_3 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_3  ≈ llk_1

    t_init!(H, P, s, H_1, P_0, s_0)

    
    llk_4 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_4  ≈ llk_1

    t_init!(H, P, s, H_1, P_0, s_0)

    
    llk_5 = kalman_likelihood_monitored(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_5  ≈ llk_1

end

@testset "Singular F matrix diagonal H" begin
    H_1 = Matrix{Float64}(I(ny))
    Z_1 = copy(Z)
    Z_1[3, :] = Z[1, :] + Z[2, :]
    Z_1[1, 1] -= sqrt(P_0[1,1])
    Z_1[2, 2] -= sqrt(P_0[2,2])
    Z_1[3, 3] -= sqrt(P_0[3,3])
    @show det(Z_1*P_0*Z_1' + H) 

    
    ws1 = KalmanLikelihoodWs(ny, ns, np, nobs)
    ws2 = KalmanFilterWs(ny, ns, np, nobs)
    P = zeros(ns, ns, nobs + 1)
    s = zeros(ns, nobs + 1)
    s[:, 1] .= s_0
    Ptt = zeros(ns, ns, nobs)
    stt = zeros(ns, nobs)
    
    t_init!(H, P, s, H_1, P_0, s_0)
    llk_1 = kalman_filter!(y, zeros(ny), Z_1, H, zeros(ns), T, R, Q, s, stt, P, Ptt, 1, nobs, 0, ws2, full_data_pattern)
    @show ws2.lik[1:5] .- ny*log(2*pi)

    P = zeros(ns, ns)
    s = zeros(ns)
    t_init!(H, P, s, H_1, P_0, s_0)
    
    llk_2 = kalman_likelihood(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @show ws1.lik[1:5]
    @test llk_2  ≈ llk_1
     
    t_init!(H, P, s, H_1, P_0, s_0)
    
    llk_3 = kalman_likelihood(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_3  ≈ llk_1
 
    t_init!(H, P, s, H_1, P_0, s_0)

    
    llk_4 = kalman_likelihood_monitored(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_4  ≈ llk_1

    t_init!(H, P, s, H_1, P_0, s_0)

    
    llk_5 = kalman_likelihood_monitored(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_5  ≈ llk_1

end

@testset "Singular F matrix Full H" begin
    Z_1 = copy(Z)
    Z_1[3, :] = Z[1, :] + Z[2, :]
    H_1 = Z_1*P_0*Z_1'
    @show det(Z_1*P_0*Z_1' + H_1) 
    
    ws1 = KalmanLikelihoodWs(ny, ns, np, nobs)
    ws2 = KalmanFilterWs(ny, ns, np, nobs)
    P = zeros(ns, ns, nobs+1)
    s = zeros(ns, nobs + 1)
    s[:, 1] .= s_0
    Ptt = zeros(ns, ns, nobs)
    stt = zeros(ns, nobs)

    t_init!(H, P, s, H_1, P_0, s_0)
    llk_1 = kalman_filter!(y, zeros(ny), Z_1, H, zeros(ns), T, R, Q, s, stt, P, Ptt, 1, nobs, 0, ws2, full_data_pattern)

    P = zeros(ns, ns)
    s = zeros(ns) 
    t_init!(H, P, s, H_1, P_0, s_0)
   
    llk_2 = kalman_likelihood(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_2  ≈ llk_1

    t_init!(H, P, s, H_1, P_0, s_0)
    
    llk_3 = kalman_likelihood(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_3  ≈ llk_1

    t_init!(H, P, s, H_1, P_0, s_0)
    
    llk_4 = kalman_likelihood_monitored(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_4  ≈ llk_1

    t_init!(H, P, s, H_1, P_0, s_0)
    
    llk_5 = kalman_likelihood_monitored(y, Z_1, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_5  ≈ llk_1

end

H = copy(H_0)    
# Fast Kalman Filter
@testset "Fast Kalman Filter" begin
    ws1 = KalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)
    ws2 = FastKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

    P = zeros(ns, ns)
    s = zeros(ns)
    t_init!(H, P, s, H_0, P_0, s_0)
    llk_1 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_2 = fast_kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws2)
    @test llk_2 ≈ llk_1

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_3 = fast_kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws2, full_data_pattern)
    @test llk_3 ≈ llk_1

end


@testset "Z as selection matrix" begin
    ws1 = KalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)
    ws2 = FastKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

    fill!(Z, 0.0)
    Z[1, 4] = 1
    Z[2, 3] = 1
    Z[3, 2] = 1
    z = [4, 3, 2]

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_1 = kalman_likelihood(y, Z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    
    t_init!(H, P, s, H_0, P_0, s_0)
    llk_2 = kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws1)
    @test llk_1 ≈ llk_2

    llk_2 = kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws1, full_data_pattern)
    @test llk_1 ≈ llk_2

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_3 = fast_kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws2)
    @test llk_1 ≈ llk_3

    t_init!(H, P, s, H_0, P_0, s_0)
    llk_3 = fast_kalman_likelihood(y, z, H, T, R, Q, s, P, 1, nobs, 0, ws2, full_data_pattern)
    @test llk_1 ≈ llk_3

end

@testset "Kalman Filter and Smoother" begin
    c = zeros(ny)
    d = zeros(ns)
    s = copy(s_0)
    stt = similar(s)
    P = copy(P_0)
    Ptt = similar(P)
    
    nobs1 = 1
    ws1 = KalmanFilterWs{Float64, Int64}(ny, ns, np, nobs1)

    kalman_filter!(y, c, Z, H, d, T, R, Q, s, stt, P, Ptt, 1, nobs1, 0, ws1, full_data_pattern)
    
    cs = zeros(ny, nobs)
    Zs = zeros(ny, ns, nobs)
    Hs = zeros(ny, ny, nobs)
    ds = zeros(ns, nobs)
    Ts = zeros(ns, ns, nobs)
    Rs = zeros(ns, np, nobs)
    Qs = zeros(np, np, nobs)
    ss = zeros(ns, nobs + 1)
    stt = zeros(ns, nobs)
    Ps = zeros(ns, ns, nobs + 1)
    Ptt = zeros(ns, ns, nobs)

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
    kalman_filter!(y, cs, Zs, Hs, ds, Ts, Rs, Qs, ss, stt, Ps, Ptt, 1, nobs1, 0, ws1, full_data_pattern)
    @test ss[:, nobs1+1] ≈ s
    @test Ps[:, : , nobs1+1] ≈ P

    ws2 = KalmanSmootherWs{Float64, Int64}(ny, ns, np, nobs)
    ss1 = zeros(ns, nobs+1)
    ss1[:, 1] = s_0
    stt = similar(ss1)
    Ps1 = similar(Ps)
    Ps1[:, :, 1] = P_0
    Ptt = similar(Ps1)
    alphah = zeros(ns, nobs)
    epsilonh = zeros(ny, nobs)
    etah = zeros(np, nobs)
    Valpha = zeros(ns, ns, nobs)
    Vepsilon = zeros(ny, ny, nobs)
    Veta = zeros(np, np, nobs)
    kalman_smoother!(y, cs, Zs, Hs, ds, Ts, Rs, Qs, ss1, stt, Ps1, Ptt, alphah, epsilonh, etah, Valpha, Vepsilon, Veta, 1, nobs, 0, ws2, full_data_pattern)
    @test ss1[:, nobs1 + 1] ≈ s
    @test Ps1[:, : , nobs1+1] ≈ P

    for i = 1:nobs
        Hs[:, :, i] = zeros(ny, ny)
    end
    kalman_smoother!(y, cs, Zs, Hs, ds, Ts, Rs, Qs, ss1, stt, Ps1, Ptt, alphah, epsilonh, etah, Valpha, Vepsilon, Veta, 1, nobs, 0, ws2, full_data_pattern)
    @test y ≈ view(alphah, [4, 3, 2], :)
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

@testset "Diffuse Kalman Filter and Smoother" begin
    ws4 = DiffuseKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)
    ws5 = DiffuseKalmanFilterWs{Float64, Int64}(ny, ns, np, nobs)
    
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    copy!(ws4.QQ, R*Q*R')
    
    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, Z, H, T,
                                                          ws4.QQ, a,
                                                          Pinf, Pstar,
                                                          1, nobs,
                                                          1e-8, ws4)
    llk_3 = -0.5*(t*ny*log(2*pi) + sum(ws4.lik[1:t]))

    # Dynare returns minus log likelihood
    @test llk_3 ≈ -vars["dLIK"]
    @test a ≈ vars["a"]
    @test Pstar ≈ vars["Pstar1"]

    c = zeros(ny)
    d = zeros(ns)
    aa = repeat(a_0, 1, nobs+1)
    att = similar(aa)
    PPinf = repeat(Pinf_0, 1, 1, nobs+1)
    Pinftt = similar(PPinf)
    PPstar = repeat(Pstar_0, 1, 1, nobs+1)
    Pstartt = similar(PPstar)
    t1 = KalmanFilterTools.diffuse_kalman_filter_init!(Y, c, Z, H, d, T, R, Q, aa, att,
                                                       PPinf, Pinftt, PPstar, Pstartt,
                                                       1, nobs, 0, 1e-8, ws5, full_data_pattern)
    @test t1 == t
    @test llk_3 ≈ -0.5*sum(ws5.lik[1:t1])
    @test aa[:, t1 + 1] ≈ vars["a"]
    @test PPstar[:, :, t1 + 1] ≈ vars["Pstar1"]
    display(PPstar[:,:,t1+1])
    display(vars["Pstar1"])
    z = [4, 3]
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, z, H, T, ws4.QQ, a, Pinf, 
                                                          Pstar, 1, nobs, 1e-8, ws4)
    llk_3 = -0.5*(t*ny*log(2*pi) + sum(ws4.lik[1:t]))

    # Dynare returns minus log likelihood
    @test llk_3 ≈ -vars["dLIK"]
    @test a ≈ vars["a"]
    @test Pstar ≈ vars["Pstar1"]

    aa = repeat(a_0, 1, nobs)
    PPinf = repeat(Pinf_0, 1, 1, nobs)
    PPstar = repeat(Pstar_0, 1, 1, nobs)
    t1 = KalmanFilterTools.diffuse_kalman_filter_init!(Y, c, z, H, d, T, R, Q, aa, att, PPinf, Pinftt,
                                                       PPstar, Pstartt, 1, nobs, 0, 1e-8, ws5, full_data_pattern)
    @test t1 == t
    @test llk_3 ≈ -0.5*sum(ws5.lik[1:t1])
    @test aa[:, t1 + 1] ≈ vars["a"]
    @test PPstar[:, :, t1 + 1] ≈ vars["Pstar1"]
    
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_4 = diffuse_kalman_likelihood(Y, Z, H, T, R, Q, a, Pinf, Pstar, 1, nobs, 0, 1e-8, ws4)
    
    aa = repeat(a_0, 1, nobs + 1)
    Pinf = copy(Pinf_0)
    Pinftt = similar(Pinf)
    Pstar = copy(Pstar_0)
    Pstartt = similar(Pstar)
    
    llk_4a = diffuse_kalman_filter!(Y, c, Z, H, d, T, R, Q, aa, att, Pinf, Pinftt, Pstar, Pstartt,
                                    1, nobs, 0, 1e-8, ws5)
    @test llk_4a ≈ llk_4
   
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_5 = diffuse_kalman_likelihood(Y, z, H, T, R, Q, a, Pinf, Pstar, 1, nobs, 0, 1e-8, ws4)
    @test llk_5 ≈ llk_4 

    aa = repeat(a_0, 1, nobs + 1)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    llk_5a = diffuse_kalman_filter!(Y, c, z, H, d, T, R, Q, aa, att, Pinf, Pinftt, Pstar, Pstartt, 1, nobs, 0, 1e-8, ws5)
    @test llk_5a ≈ llk_5
    
    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)

    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, Z, H, T, ws4.QQ, a, Pinf, Pstar, 1, nobs, 1e-8, ws4, full_data_pattern)
    llk_3 = -0.5*sum(ws4.lik[1:t])

    # Dynare returns minus log likelihood
    @test llk_3 ≈ -vars["dLIK"]
    @test a ≈ vars["a"]
    @test Pstar ≈ vars["Pstar1"]

    a = copy(a_0)
    Pinf = copy(Pinf_0)
    Pstar = copy(Pstar_0)
    t = KalmanFilterTools.diffuse_kalman_likelihood_init!(Y, z, H, T, ws4.QQ, a, Pinf, Pstar, 1, nobs, 1e-8, ws4, full_data_pattern)
    llk_3 = -0.5*sum(ws4.lik[1:t])

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

    ws6 = DiffuseKalmanSmootherWs{Float64, Int64}(ny, ns, np, nobs)

    aa = zeros(ns, nobs + 1)
    aa[:, 1] .= a_0
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
    llk_6a = diffuse_kalman_smoother!(Y, c, Z, H, d, T, R, Q, aa, att,
                                      Pinf, Pinftt, Pstar, Pstartt,
                                      alphah, epsilonh, etah, Valphah,
                                      Vepsilonh, Vetah, 1, nobs, 0,
                                      1e-8, ws6)
    @test llk_6a ≈ llk_4 
    @test Y ≈ alphah[z, :]

    aa = zeros(ns, nobs + 1)
    aa[:, 1] .= a_0
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
    llk_6b = diffuse_kalman_smoother!(Y, c, z, H, d, T, R, Q, aa, att,
                                      Pinf, Pinftt, Pstar, Pstartt,
                                      alphah, epsilonh, etah, Valphah,
                                      Vepsilonh, Vetah, 1, nobs, 0,
                                      1e-8, ws6)
    @test llk_6b ≈ llk_4 
    @test Y ≈ alphah[z, :]
end

@testset "start and last" begin
    nobs1 = nobs - 1
    ws1 = KalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

    P_0 = randn(ns, ns)
    P_0 = P_0'*P_0
    
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

    ws4 = DiffuseKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)
    
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

@testset "smoother" begin
end    

nothing

