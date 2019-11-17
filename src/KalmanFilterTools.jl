module KalmanFilterTools

include("kalman_base.jl")

"""
State space specification:
    y_t = Z*a_t + epsilon_t
    a_t = T*a_{t-1} + R eta_t
    E(epsilon_t epsilon_t') = H
    E(eta_t eta_t') = Q
"""

using LinearAlgebra
using LinearAlgebra.BLAS

export KalmanLikelihoodWs, FastKalmanLikelihoodWs, DiffuseKalmanLikelihoodWs, KalmanSmootherWs, kalman_likelihood, kalman_likelihood_monitored, fast_kalman_likelihood, diffuse_kalman_likelihood, kalman_filter!

abstract type KalmanWs{T, U} end


struct KalmanLikelihoodWs{T, U} <: KalmanWs{T, U}
    Zsmall::Matrix{T}
    # necessary for Z selecting vector with missing variables
    iZsmall::Vector{U}
    RQ::Matrix{T}
    QQ::Matrix{T}
    v::Vector{T}
    F::Matrix{T}
    cholF::Matrix{T}
    iFv::Vector{T}
    a1::Vector{T}
    K::Matrix{T}
    ZP::Matrix{T}
    iFZ::SubArray{T}  
    PTmp::Matrix{T}
    oldP::Matrix{T}
    lik::Vector{T}
    
    function KalmanLikelihoodWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        RQ = Matrix{T}(undef, ns, np)
        QQ = Matrix{T}(undef, ns, ns)
        F = Matrix{T}(undef, ny, ny)
        cholF = Matrix{T}(undef, ny, ny)
        v = Vector{T}(undef, ny)
        iFv = Vector{T}(undef, ny)
        a1 = Vector{T}(undef, ns)
        K = Matrix{T}(undef, ny, ns)
        PTmp = Matrix{T}(undef, ns, ns)
        oldP = Matrix{T}(undef, ns, ns)
        ZP = Matrix{T}(undef, ny, ns)
        iFZ = view(PTmp,1:ny,:)
        lik = Vector{T}(undef, nobs)
        new(Zsmall, iZsmall, RQ, QQ, v, F, cholF, iFv, a1, K, ZP, iFZ, PTmp, oldP, lik)
    end
end

# Z can be either a matrix or a selection vector
function kalman_likelihood(Y::AbstractArray{U},
                           Z::AbstractArray{W},
                           H::AbstractArray{U},
                           T::AbstractArray{U},
                           R::AbstractArray{U},
                           Q::AbstractArray{U},
                           a::AbstractArray{U},
                           P::AbstractArray{U},
                           start::V,
                           last::V,
                           presample::V,
                           ws::KalmanWs) where {U <: AbstractFloat, W <: Real, V <: Integer}
    ny, nobs = size(Y)
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = 1
    @inbounds while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        # F  = Z*P*Z' + H
        get_F!(ws.F, ws.ZP, Z, P, H)
        get_cholF!(ws.cholF, ws.F)
        # iFv = inv(F)*v
        get_iFv!(ws.iFv, ws.cholF, ws.v)
        ws.lik[t] = log(det_from_cholesky(ws.cholF)) + LinearAlgebra.dot(ws.v, ws.iFv)
        if t < last
            # K = iF*Z*P
            get_K!(ws.K, ws.ZP, ws.cholF) 
            # a = T(a + K'*v)
            update_a!(a, ws.K, ws.v, ws.a1, T)
            # P = T*(P - K'*Z*P)*T'+ QQ
            update_P!(P, T, ws.QQ, ws.K, ws.ZP, ws.PTmp)
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*(lik_cst + sum(view(ws.lik,(presample+1):nobs)))
    else
        LIK = -0.5*(lik_cst + sum(ws.lik))
    end
    LIK
end

function get_vZsmall(ws::KalmanWs, Z::Matrix{T}, pattern::Vector{U}, n::U, ny::U) where {T <: AbstractFloat, U <: Integer}
    n = length(pattern)
    vZsmall = view(ws.Zsmall, 1:n, :)
    if n == ny
        copyto!(vZsmall, Z)
    else
        vZsmall .= view(Z, pattern, :)
    end
end

function get_vZsmall(ws::KalmanWs, Z::Vector{U}, pattern::Vector{U}, n::U, ny::U) where {T <: AbstractFloat, U <: Integer}
    n = length(pattern)
    vZsmall = view(ws.iZsmall, 1:n)
    if n == ny
        copyto!(vZsmall, Z)
    else
        vZsmall .= view(Z, pattern)
    end
end

function kalman_likelihood(Y::AbstractArray{U},
                           Z::AbstractArray{W},
                           H::AbstractArray{U},
                           T::AbstractArray{U},
                           R::AbstractArray{U},
                           Q::AbstractArray{U},
                           a::AbstractArray{U},
                           P::AbstractArray{U},
                           start::V,
                           last::V,
                           presample::V,
                           ws::KalmanWs,
                           data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat, W <: Real, V <: Integer}
    ny, nobs = size(Y)
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = 1
    ncolZ = size(Z, 2)
    @inbounds while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vH = view(H, pattern, pattern)
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata)
        vK = view(ws.K, 1:ndata, :)
        vZsmall = get_vZsmall(ws, Z, pattern, ndata, ny)
        
        # v  = Y[:,t] - Z*a
        get_v!(vv, Y, vZsmall, a, t, pattern)
        iy += ny
        # F  = Z*P*Z' + H
        get_F!(vF, vZP, vZsmall, P, vH)
        get_cholF!(vcholF, vF)
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
        if t < last
            # K = iF*Z*P
            get_K!(vK, vZP, vcholF) 
            # a = T(a + K'*v)
            update_a!(a, vK, vv, ws.a1, T)
            # P = T*(P - K'*Z*P)*T'+ QQ
            update_P!(P, T, ws.QQ, vK, vZP, ws.PTmp)
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*sum(view(ws.lik,(presample+1):nobs))
    else
        LIK = -0.5*sum(ws.lik)
    end
    LIK
end

function kalman_likelihood_monitored(Y::Matrix{U},
                                     Z::AbstractArray{W},
                                     H::Matrix{U},
                                     T::Matrix{U},
                                     R::Matrix{U},
                                     Q::Matrix{U},
                                     a::Vector{U},
                                     P::Matrix{U},
                                     start::V,
                                     last::V,
                                     presample::V,
                                     ws::KalmanWs) where {U <: AbstractFloat, V <: Integer, W <: Real}
    ny, nobs = size(Y)
    ns = size(T,1)
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = 1
    steady = false
    copy!(ws.oldP, P)
    @inbounds while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        if !steady
            # F  = Z*P*Z' + H
            get_F!(ws.F, ws.ZP, Z, P, H)
            get_cholF!(ws.cholF, ws.F)
        end
        # iFv = inv(F)*v
        get_iFv!(ws.iFv, ws.cholF, ws.v)
        ws.lik[t] = log(det_from_cholesky(ws.cholF)) + LinearAlgebra.dot(ws.v, ws.iFv)
        if t < last
            if !steady
                # K = iF*Z*P
                get_K!(ws.K, ws.ZP, ws.cholF)
            end
            # a = T(a + K'*v)
            update_a!(a, ws.K, ws.v, ws.a1, T)
            if !steady
                # P = T*(P - K'*Z*P)*T'+ QQ
                update_P!(P, T, ws.QQ, ws.K, ws.ZP, ws.PTmp)
                ws.oldP .-= P
                if norm(ws.oldP) < ns*eps()
                    steady = true
                else
                    copy!(ws.oldP, P)
                end
            end
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*(lik_cst + sum(view(ws.lik,(presample+1):nobs)))
    else
        LIK = -0.5*(lik_cst + sum(ws.lik))
    end
    LIK
end

function kalman_likelihood_monitored(Y::AbstractArray{U},
                                     Z::AbstractArray{W},
                                     H::AbstractArray{U},
                                     T::AbstractArray{U},
                                     R::AbstractArray{U},
                                     Q::AbstractArray{U},
                                     a::AbstractArray{U},
                                     P::AbstractArray{U},
                                     start::V,
                                     last::V,
                                     presample::V,
                                     ws::KalmanWs,
                                     data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat, W <: Real, V <: Integer}
    ny, nobs = size(Y)
    ns = size(T,1)
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = 1
    steady = false
    copy!(ws.oldP, P)
    @inbounds while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vH = view(H, pattern, pattern)
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata)
        vK = view(ws.K, 1:ndata, :)
        
        # v  = Y[:,t] - Z*a
        get_v!(vv, Y, Z, a, t, pattern)
        iy += ny
        if !steady
            # F  = Z*P*Z' + H
            get_F!(vF, vZP, Z, P, vH)
            get_cholF!(vcholF, vF)
        end
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
        if t < last
            if !steady
                # K = iF*Z*P
                get_K!(vK, vZP, vcholF)
            end
            # a = T(a + K'*v)
            update_a!(a, vK, vv, ws.a1, T)
            if !steady
                # P = T*(P - K'*Z*P)*T'+ QQ
                update_P!(P, T, ws.QQ, vK, vZP, ws.PTmp)
                ws.oldP .-= P
                if norm(ws.oldP) < ns*eps()
                    steady = true
                else
                    copy!(ws.oldP, P)
                end
            end
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*(sum(view(ws.lik,(presample+1):nobs)))
    else
        LIK = -0.5*(sum(ws.lik))
    end
    LIK
end

struct FastKalmanLikelihoodWs{T, U} <: KalmanWs{T, U}
    Zsmall::Matrix{T}
    iZsmall::Vector{U}
    QQ::Matrix{T}
    v::Vector{T}
    F::Matrix{T}
    cholF::Matrix{T}
    iFv::Vector{T}
    a1::Vector{T}
    K::Matrix{T}
    RQ::Matrix{T}
    ZP::Matrix{T}
    M::Matrix{T}
    W::Matrix{T}
    ZW::Matrix{T}
    ZWM::Matrix{T}
    iFZWM::Matrix{T}
    TW::Matrix{T}
    iFZW::Matrix{T}
    KtiFZW::Matrix{T}
    lik::Vector{T}
    
    function FastKalmanLikelihoodWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        QQ = Matrix{T}(undef, ns, ns)
        RQ = Matrix{T}(undef, ns, np)
        F = Matrix{T}(undef, ny, ny)
        cholF = Matrix{T}(undef, ny, ny)
        v = Vector{T}(undef, ny)
        
        iFv = Vector{T}(undef, ny)
        a1 = Vector{T}(undef, ns)
        K = Matrix{T}(undef, ny, ns)
        M = Matrix{T}(undef, ny, ny)
        W = Matrix{T}(undef, ns, ny)
        ZP = Matrix{T}(undef, ny, ns)
        ZW = Matrix{T}(undef, ny, ny)
        ZWM = Matrix{T}(undef, ny, ny)
        iFZWM = Matrix{T}(undef, ny, ny)
        TW = Matrix{T}(undef, ns, ny)
        iFZW = Matrix{T}(undef, ny, ny)
        KtiFZW = Matrix{T}(undef, ns, ny)
        lik = Vector{T}(undef, nobs)
        new(Zsmall, iZsmall, QQ, v, F, cholF, iFv, a1, K, RQ, ZP, M, W, ZW, ZWM, iFZWM, TW, iFZW, KtiFZW, lik)
    end
end

"""
from kalman_filter_2
K doesn't represent the same matrix as above
"""
function fast_kalman_likelihood(Y::Matrix{U},
                                Z::AbstractArray{W},
                                H::Matrix{U},
                                T::Matrix{U},
                                R::Matrix{U},
                                Q::Matrix{U},
                                a::Vector{U},
                                P::Matrix{U},
                                start::V,
                                last::V,
                                presample::V,
                                ws::KalmanWs) where {U <: AbstractFloat, V <: Integer, W <: Real}
    ny, nobs = size(Y)
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    fill!(ws.lik, 0.0)
    # F  = Z*P*Z' + H
    get_F!(ws.F, ws.ZP, Z, P, H)
    get_cholF!(ws.cholF, ws.F)
    # K = Z*P
    copy!(ws.K, ws.ZP)
    # W = T*K'
    mul!(ws.W, T, transpose(ws.K))
    # M = -iF
    get_M!(ws.M, ws.cholF, ws.ZW) 
    LIK = 0
    t = start
    iy = 1
    @inbounds while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        # iFv = inv(F)*v
        get_iFv!(ws.iFv, ws.cholF, ws.v)
        ws.lik[t] = log(det_from_cholesky(ws.cholF)) + LinearAlgebra.dot(ws.v, ws.iFv)
        if t < last
            # a = T(a + K'*iFv)
            update_a!(a, ws.K, ws.iFv, ws.a1, T)
            # M = M + M*W'*Z'iF*Z*W*M
            update_M!(ws.M, Z, ws.W, ws.cholF, ws.ZW, ws.ZWM, ws.iFZWM)
            # F =  F + Z*W*M*W'Z'
            gemm!('N', 'T', 1.0, ws.ZWM, ws.ZW, 1.0, ws.F)
            # cholF
            get_cholF!(ws.cholF, ws.F)
            # K = K + W*M*W'*Z'
            update_K!(ws.K, ws.ZWM, ws.W)
            # W = T(W - K'*iF*Z*W)
            update_W!(ws.W, ws.ZW, ws.cholF, T, ws.K, ws.iFZW, ws.KtiFZW)
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*(lik_cst + sum(view(ws.lik, (presample+1):nobs)))
    else
        LIK = -0.5*(lik_cst + sum(ws.lik))
    end
    LIK
end

function fast_kalman_likelihood(Y::Matrix{U},
                                Z::AbstractArray{W},
                                H::Matrix{U},
                                T::Matrix{U},
                                R::Matrix{U},
                                Q::Matrix{U},
                                a::Vector{U},
                                P::Matrix{U},
                                start::V,
                                last::V,
                                presample::V,
                                ws::KalmanWs,
                                data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat, V <: Integer, W <: Real}
    ny, nobs = size(Y)
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    # F  = Z*P*Z' + H
    get_F!(ws.F, ws.ZP, Z, P, H)
    get_cholF!(ws.cholF, ws.F)
    # K = Z*P
    copy!(ws.K, ws.ZP)
    # W = T*K'
    mul!(ws.W, T, transpose(ws.K))
    # M = -iF
    get_M!(ws.M, ws.cholF, ws.ZW) 
    LIK = 0
    t = start
    iy = 1
    @inbounds while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata)
        vK = view(ws.K, 1:ndata, :)
        vW = view(ws.W, :, 1:ndata)
        vZW = view(ws.ZW, 1:ndata, 1:ndata)
        viFZW = view(ws.iFZW, 1:ndata, 1:ndata)
        vZWM = view(ws.ZWM, 1:ndata, 1:ndata)
        viFZWM = view(ws.iFZWM, 1:ndata, 1:ndata)
        vKtiFZW = view(ws.KtiFZW, :, 1:ndata)
        vZsmall = get_vZsmall(ws, Z, pattern, ndata, ny)
        
        # v  = Y[:,t] - Z*a
        get_v!(vv, Y, vZsmall, a, t, pattern)
        iy += ny
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
        if t < last
            # a = T(a + K'*iFv)
            update_a!(a, vK, viFv, ws.a1, T)
            # M = M + M*W'*Z'iF*Z*W*M
            update_M!(ws.M, Z, vW, vcholF, vZW, vZWM, viFZWM)
            # F =  F + Z*W*M*W'Z'
            gemm!('N', 'T', 1.0, vZWM, vZW, 1.0, vF)
            # cholF
            get_cholF!(vcholF, vF)
            # K = K + W*M*W'*Z'
            update_K!(vK, vZWM, vW)
            # W = T(W - K'*iF*Z*W)
            update_W!(vW, vZW, vcholF, T, vK, viFZW, vKtiFZW)
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*(sum(view(ws.lik, (presample+1):nobs)))
    else
        LIK = -0.5*(sum(ws.lik))
    end
    LIK
end

struct DiffuseKalmanLikelihoodWs{T, U} <: KalmanWs{T, U}
    Zsmall::Matrix{T}
    iZsmall::Vector{U}
    QQ::Matrix{T}
    RQ::Matrix{T}
    v::Vector{T}
    F::Matrix{T}
    iF::Matrix{T}
    iFv::Vector{T}
    a1::Vector{T}
    cholF::Matrix{T}
    ZP::Matrix{T}
    Fstar::Matrix{T}
    ZPstar::Matrix{T}
    K::Matrix{T}
    iFZ::Matrix{T}
    Kstar::Matrix{T}
    PTmp::Matrix{T}
    uKinf::Vector{T}
    uKstar::Vector{T}
    Kinf_Finf::Vector{T}
    lik::Vector{T}
    function DiffuseKalmanLikelihoodWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        QQ = Matrix{T}(undef, ns, ns)
        RQ = Matrix{T}(undef, ns, np)
        v = Vector{T}(undef, ny)
        F = Matrix{T}(undef, ny, ny)
        iF = Matrix{T}(undef, ny,ny )
        iFv = Vector{T}(undef, ny)
        a1 = Vector{T}(undef, ns)
        cholF = Matrix{T}(undef, ny, ny)
        ZP = Matrix{T}(undef, ny, ns)
        Fstar = Matrix{T}(undef, ny, ny)
        ZPstar = Matrix{T}(undef, ny, ns)
        K = Matrix{T}(undef, ny, ns)
        iFZ = Matrix{T}(undef, ny, ns)
        Kstar = Matrix{T}(undef, ny, ns)
        PTmp = Matrix{T}(undef, ns, ns)
        uKinf = Vector{T}(undef, ns)
        uKstar = Vector{T}(undef, ns)
        Kinf_Finf = Vector{T}(undef, ns)
        lik = zeros(T, nobs)
        new(Zsmall, iZsmall, QQ, RQ, v, F, iF, iFv, a1, cholF, ZP, Fstar, ZPstar, K, iFZ, Kstar, PTmp, uKinf, uKstar, Kinf_Finf, lik)
    end
end

function diffuse_kalman_likelihood_init!(Y::Matrix{U},
                                         Z::AbstractArray{W},
                                         H::Matrix{U},
                                         T::Matrix{U},
                                         QQ::Matrix{U},
                                         a::Vector{U},
                                         Pinf::Matrix{U},
                                         Pstar::Matrix{U},
                                         start::V,
                                         last::V,
                                         tol::U,
                                         ws::KalmanWs) where {U <: AbstractFloat,
                                                              V <: Integer,
                                                              W <: Real}
    
    ny = size(Y, 1)
    t = start
    LIK = 0
    iy = 1
    diffuse_kalman_tol = 1e-8
    kalman_tol = 1e-8
    while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        # Finf = Z*Pinf*Z'
        get_F!(ws.F, ws.ZP, Z, Pinf)
        info = get_cholF!(ws.cholF, ws.F)
        if info[2] > 0
            if norm(ws.F) < tol
                return t - 1
            else
                ws.lik[t] += univariate_step(t, Y, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws)
            end
        else
            ws.lik[t] = log(det_from_cholesky(ws.cholF))
            # Kinf   = iFinf*Z*Pinf                                   %define Kinf'=T^{-1}*K_0 with M_{\infty}=Pinf*Z'
            copy!(ws.K, ws.ZP)
            LAPACK.potrs!('U', ws.cholF, ws.K)
            # Fstar  = Z*Pstar*Z' + H;                                        %(5.7) DK(2012)
            get_F!(ws.Fstar, ws.ZPstar, Z, Pstar, H)
            # Kstar  = iFinf*(Z*Pstar - Fstar*Kinf)                           %(5.12) DK(2012); note that there is a typo in DK (2003) with "+ Kinf" instead of "- Kinf", but it is correct in their appendix
            get_Kstar!(ws.Kstar, Z, Pstar, ws.Fstar, ws.K, ws.cholF)
            # Pstar  = T*(Pstar-Pstar*Z'*Kinf-Pinf*Z'*Kstar)*T'+QQ;         %(5.14) DK(2012)
            copy!(ws.PTmp, Pstar)
            gemm!('T','N',-1.0,ws.ZPstar, ws.K, 1.0, ws.PTmp)
            gemm!('T','N',-1.0,ws.ZP, ws.Kstar, 1.0, ws.PTmp)
            copy!(Pstar, ws.PTmp)
            mul!(ws.PTmp,T,Pstar)
            copy!(Pstar, QQ)
            gemm!('N','T',1.0,ws.PTmp,T,1.0,Pstar)
            # Pinf   = T*(Pinf-Pinf*Z'*Kinf)*T';                             %(5.14) DK(2012)
            gemm!('T','N', -1.0,ws.ZP, ws.K,1.0,Pinf)
            mul!(ws.PTmp,T,Pinf)
            mul!(Pinf,ws.PTmp,transpose(T))
            # a      = T*(a+Kinf*v);                                          %(5.13) DK(2012)
            update_a!(a, ws.K, ws.v, ws.a1, T)
        end
        t += 1
    end
    t
end

function diffuse_kalman_likelihood_init!(Y::Matrix{U},
                                         Z::AbstractArray{W},
                                         H::Matrix{U},
                                         T::Matrix{U},
                                         QQ::Matrix{U},
                                         a::Vector{U},
                                         Pinf::Matrix{U},
                                         Pstar::Matrix{U},
                                         start::V,
                                         last::V,
                                         tol::U,
                                         ws::KalmanWs,
                                         data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat,
                                                                                 V <: Integer,
                                                                                 W <: Real}
    
    ny = size(Y, 1)
    t = start
    LIK = 0
    iy = 1
    diffuse_kalman_tol = 1e-8
    kalman_tol = 1e-8
    while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vH = view(H, pattern, pattern)
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vFstar = view(ws.Fstar, 1:ndata, 1:ndata)
        vZP = view(ws.ZP, 1:ndata, :)
        vZPstar = view(ws.ZPstar, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata)
        vK = view(ws.K, 1:ndata, :)
        vKstar = view(ws.Kstar, 1:ndata, :)
        vZsmall = get_vZsmall(ws, Z, pattern, ndata, ny)
        
        # v  = Y[:,t] - Z*a
        get_v!(vv, Y, vZsmall, a, t, pattern)
        iy += ny
        # Finf = Z*Pinf*Z'
        get_F!(vF, vZP, vZsmall, Pinf)
        info = get_cholF!(vcholF, vF)
        if info[2] > 0
            if norm(vF) < tol
                return t - 1
            else
                ws.lik[t] += univariate_step(t, Y, vZsmall, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, pattern, ws)
            end
        else
            ws.lik[t] = log(det_from_cholesky(ws.cholF))
            # Kinf   = iFinf*Z*Pinf                                   %define Kinf'=T^{-1}*K_0 with M_{\infty}=Pinf*Z'
            copy!(vK, vZP)
            LAPACK.potrs!('U', vcholF, vK)
            # Fstar  = Z*Pstar*Z' + H;                                        %(5.7) DK(2012)
            get_F!(vFstar, vZPstar, vZsmall, Pstar, vH)
            # Kstar  = iFinf*(Z*Pstar - Fstar*Kinf)                           %(5.12) DK(2012); note that there is a typo in DK (2003) with "+ Kinf" instead of "- Kinf", but it is correct in their appendix
            get_Kstar!(vKstar, Z, Pstar, vFstar, vK, vcholF)
            # Pstar  = T*(Pstar-Pstar*Z'*Kinf-Pinf*Z'*Kstar)*T'+QQ;         %(5.14) DK(2012)
            copy!(ws.PTmp, Pstar)
            gemm!('T','N',-1.0, vZPstar, vK, 1.0, ws.PTmp)
            gemm!('T','N',-1.0, vZP, vKstar, 1.0, ws.PTmp)
            copy!(Pstar, ws.PTmp)
            mul!(ws.PTmp,T,Pstar)
            copy!(Pstar, QQ)
            gemm!('N','T',1.0,ws.PTmp,T,1.0,Pstar)
            # Pinf   = T*(Pinf-Pinf*Z'*Kinf)*T';                             %(5.14) DK(2012)
            gemm!('T','N', -1.0, vZP, ws.K,1.0,Pinf)
            mul!(ws.PTmp,T,Pinf)
            mul!(Pinf,ws.PTmp,transpose(T))
            # a      = T*(a+Kinf*v);                                          %(5.13) DK(2012)
            update_a!(a, vK, vv, ws.a1, T)
        end
        t += 1
    end
    t
end

function diffuse_kalman_likelihood(Y::Matrix{U},
                                   Z::AbstractArray{W},
                                   H::Matrix{U},
                                   T::Matrix{U},
                                   R::Matrix{U},     
                                   Q::Matrix{U},
                                   a::Vector{U},
                                   Pinf::Matrix{U},
                                   Pstar::Matrix{U},
                                   start::V,
                                   last::V,
                                   presample::V,
                                   tol::U,
                                   ws::DiffuseKalmanLikelihoodWs) where {U <: AbstractFloat,
                                                                         V <: Integer,
                                                                         W <: Real}
    ny, nobs = size(Y)
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    fill!(ws.lik, 0.0)
    t = diffuse_kalman_likelihood_init!(Y, Z, H, T, ws.QQ, a, Pinf, Pstar, start, last, tol, ws)
    kalman_likelihood(Y, Z, H, T, R, Q, a, Pstar, t, last, presample, ws)
    @inbounds if presample > 0
        LIK = -0.5*(lik_cst + sum(view(ws.lik, (presample+1):nobs)))
    else
        LIK = -0.5*(lik_cst + sum(ws.lik))
    end
    LIK
end

function diffuse_kalman_likelihood(Y::Matrix{U},
                                   Z::AbstractArray{W},
                                   H::Matrix{U},
                                   T::Matrix{U},
                                   R::Matrix{U},     
                                   Q::Matrix{U},
                                   a::Vector{U},
                                   Pinf::Matrix{U},
                                   Pstar::Matrix{U},
                                   start::V,
                                   last::V,
                                   presample::V,
                                   tol::U,
                                   ws::DiffuseKalmanLikelihoodWs,
                                   data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat,
                                                                         V <: Integer,
                                                                         W <: Real}
    ny, nobs = size(Y)
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    fill!(ws.lik, 0.0)
    t = diffuse_kalman_likelihood_init!(Y, Z, H, T, ws.QQ, a, Pinf, Pstar, start, last, tol, ws, data_pattern)
    kalman_likelihood(Y, Z, H, T, R, Q, a, Pstar, t, last, presample, ws, data_pattern)
    @inbounds if presample > 0
        LIK = -0.5*(lik_cst + sum(view(ws.lik, (presample+1):nobs)))
    else
        LIK = -0.5*(lik_cst + sum(ws.lik))
    end
    LIK
end


function univariate_step(t, Y, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws)
    llik = 0
    for i=1:size(Y, 1)
        Zi = view(Z, i, :)
        prediction_error = get_prediction_error(Y, Z, a, i, t)
        Fstar = get_Fstar!(Zi, Pstar, H[i], ws.uKstar)
        Finf = get_Finf!(Zi, Pstar, ws.uKstar)
        # Conduct check of rank
        # Pinf and Finf are always scaled such that their norm=1: Fstar/Pstar, instead,
        # depends on the actual values of std errors in the model and can be badly scaled.
        # experience is that diffuse_kalman_tol has to be bigger than kalman_tol, to ensure
        # exiting the diffuse filter properly, avoiding tests that provide false non-zero rank for Pinf.
        # Also the test for singularity is better set coarser for Finf than for Fstar for the same reason
        if Finf > diffuse_kalman_tol                 # F_{\infty,t,i} = 0, use upper part of bracket on p. 175 DK (2012) for w_{t,i}
            ws.Kinf_Finf .= ws.uKinf./Finf
            a .+= prediction_error.*ws.Kinf_Finf
            # Pstar     = Pstar + Kinf*(Kinf_Finf'*(Fstar/Finf)) - Kstar*Kinf_Finf' - Kinf_Finf*Kstar'
            ger!( Fstar/Finf, ws.uKinf, ws.Kinf_Finf, Pstar) 
            ger!( -1.0, ws.uKstar, ws.Kinf_Finf, Pstar) 
            ger!( -1.0, ws.Kinf_Finf, ws.uKstar, Pstar) 
            # Pinf      = Pinf - Kinf*Kinf_Finf'
            ger!(-1.0, ws.uKinf, ws.Kinf_Finf, Pinf) 
            llik += log(Finf)
        elseif Fstar > kalman_tol
            llik += log(Fstar) + prediction_error*prediction_error/Fstar
            a .+= ws.uKstar.*(prediction_error/Fstar)
            ger!(-1/Fstar, ws.uKstar, ws.uKstar, Pstar)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
    end
    return llik
end

function univariate_step(t, Y, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, pattern, ws)
    llik = 0
    for i=1:size(pattern, 1)
        Zi = view(Z, pattern[i], :)
        prediction_error = get_prediction_error(Y, Z, a, pattern[i], t)
        Fstar = get_Fstar!(Zi, Pstar, H[i], ws.uKstar)
        Finf = get_Finf!(Zi, Pstar, ws.uKstar)
        # Conduct check of rank
        # Pinf and Finf are always scaled such that their norm=1: Fstar/Pstar, instead,
        # depends on the actual values of std errors in the model and can be badly scaled.
        # experience is that diffuse_kalman_tol has to be bigger than kalman_tol, to ensure
        # exiting the diffuse filter properly, avoiding tests that provide false non-zero rank for Pinf.
        # Also the test for singularity is better set coarser for Finf than for Fstar for the same reason
        if Finf > diffuse_kalman_tol                 # F_{\infty,t,i} = 0, use upper part of bracket on p. 175 DK (2012) for w_{t,i}
            ws.Kinf_Finf .= ws.uKinf./Finf
            a .+= prediction_error.*ws.Kinf_Finf
            # Pstar     = Pstar + Kinf*(Kinf_Finf'*(Fstar/Finf)) - Kstar*Kinf_Finf' - Kinf_Finf*Kstar'
            ger!( Fstar/Finf, ws.uKinf, ws.Kinf_Finf, Pstar) 
            ger!( -1.0, ws.uKstar, ws.Kinf_Finf, Pstar) 
            ger!( -1.0, ws.Kinf_Finf, ws.uKstar, Pstar) 
            # Pinf      = Pinf - Kinf*Kinf_Finf'
            ger!(-1.0, ws.uKinf, ws.Kinf_Finf, Pinf) 
            llik += log(Finf)
        elseif Fstar > kalman_tol
            llik += log(Fstar) + prediction_error*prediction_error/Fstar
            a .+= ws.uKstar.*(prediction_error/Fstar)
            ger!(-1/Fstar, ws.uKstar, ws.uKstar, Pstar)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
    end
    return llik
end

# Filters

function kalman_filter_1!(Y::AbstractArray{U},
                        c::AbstractArray{U},
                        Z::AbstractArray{W},
                        H::AbstractArray{U},
                        d::AbstractArray{U},
                        T::AbstractArray{U},
                        R::AbstractArray{U},
                        Q::AbstractArray{U},
                        a::AbstractArray{U},
                        P::AbstractArray{U},
                       start::V,
                       last::V,
                       presample::V,
                       ws::KalmanWs,
                       data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat, W <: Real, V <: Integer}
    changeC = ndims(c) > 1
    changeZ = ndims(Z) > 2
    changeH = ndims(H) > 2
    changeD = ndims(d) > 1
    changeT = ndims(T) > 2
    changeR = ndims(R) > 2
    changeQ = ndims(Q) > 2
    changeA = ndims(a) > 1
    changeP = ndims(P) > 2
    
    ny, nobs = size(Y)
    ns = size(T,1)
    # QQ = R*Q*R'
    vR = view(R, :, :, 1)
    vQ = view(Q, :, :, 1)
    get_QQ!(ws.QQ, vR, vQ, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = 1
    steady = false
    vP = view(P, :, :, 1)
    copy!(ws.oldP, vP)
    @inbounds while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, pattern, t) : view(c, pattern)
        vZ = changeZ ? view(Z, :, :, t) : view(Z, :, :)
        vH = changeH ? view(H, :, :, t) : view(H, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vQ = changeR ? view(Q, :, :, t) : view(Q, :, :)
        va = changeR ? view(a, :, t) : view(a, :)
        va1 = changeA ? view(a, :, t + 1) : view(a, :)
        vd = changeD ? view(d, :, t) : view(d, :)
        vP = changeP ? view(P, :, :, t) : view(P, :, :)
        vP1 = changeP ? view(P, :, :, t + 1) : view(P, :, :)
        if changeR || changeQ
            get_QQ!(ws.QQ, vR, vQ, ws.RQ)
        end
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata)
        vK = view(ws.K, 1:ndata, :)
        
        # v  = Y[:,t] - c - Z*a
        get_v!(vv, Y, vc, vZ, va, t, pattern)
        iy += ny
        if !steady
            # F  = Z*P*Z' + H
            get_F!(vF, vZP, vZ, vP, vH)
            get_cholF!(vcholF, vF)
        end
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
        if t <= last
            if !steady
                # K = iF*Z*P
                get_K!(vK, vZP, vcholF)
            end
            # a = d + T(a + K'*v)
            update_a!(va1, va, vd, vK, vv, ws.a1, vT)
            if !steady
                # P = T*(P - K'*Z*P)*T'+ QQ
                copy!(vP1, vP)
                update_P!(vP1, vT, ws.QQ, vK, vZP, ws.PTmp)
                ws.oldP .-= vP
                if norm(ws.oldP) < ns*eps()
                    steady = true
                else
                    copy!(ws.oldP, vP)
                end
            end
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*(sum(view(ws.lik,(presample+1):nobs)))
    else
        LIK = -0.5*(sum(ws.lik))
    end
    LIK
end

struct KalmanSmootherWs{T, U} <: KalmanWs{T, U}
    Zsmall::Matrix{T}
    # necessary for Z selecting vector with missing variables
    iZsmall::Vector{U}
    RQ::Matrix{T}
    QQ::Matrix{T}
    v::Vector{T}
    F::Matrix{T}
    cholF::Matrix{T}
    iFv::Matrix{T}
    r::Vector{T}
    a1::Vector{T}
    K::Matrix{T}
    L::Array{T}
    L1::Matrix{T}
    N::Matrix{T}
    ZP::Matrix{T}
    B::Array{T}
    Kv::Matrix{T}
    iFZ::SubArray{T}  
    PTmp::Matrix{T}
    oldP::Matrix{T}
    lik::Vector{T}
    
    function KalmanSmootherWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        RQ = Matrix{T}(undef, ns, np)
        QQ = Matrix{T}(undef, ns, ns)
        F = Matrix{T}(undef, ny, ny)
        cholF = Matrix{T}(undef, ny, ny)
        v = Vector{T}(undef, ny)
        iFv = Matrix{T}(undef, ny, nobs)
        r = Vector{T}(undef, ns)
        a1 = Vector{T}(undef, ns)
        K = Matrix{T}(undef, ny, ns)
        L = Array{T}(undef, ns, ns, nobs)
        L1 = Matrix{T}(undef, ns, ns)
        N = Matrix{T}(undef, ns, ns)
        Kv = Matrix{T}(undef, ns, nobs)
        B = Array{T}(undef, ns, ns, nobs)
        PTmp = Matrix{T}(undef, ns, ns)
        oldP = Matrix{T}(undef, ns, ns)
        ZP = Matrix{T}(undef, ny, ns)
        iFZ = view(PTmp,1:ny,:)
        lik = Vector{T}(undef, nobs)
        new(Zsmall, iZsmall, RQ, QQ, v, F, cholF, iFv, r, a1, K, L, L1, N, ZP, B, Kv, iFZ, PTmp, oldP, lik)
    end
end

function kalman_filter_2!(Y::AbstractArray{U},
                          c::AbstractArray{U},
                          Z::AbstractArray{W},
                          H::AbstractArray{U},
                          d::AbstractArray{U},
                          T::AbstractArray{U},
                          R::AbstractArray{U},
                          Q::AbstractArray{U},
                          a::AbstractArray{U},
                          P::AbstractArray{U},
                          start::V,
                          last::V,
                          presample::V,
                          ws::KalmanSmootherWs,
                          data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat, W <: Real, V <: Integer}
    changeC = ndims(c) > 1
    changeZ = ndims(Z) > 2
    changeH = ndims(H) > 2
    changeD = ndims(d) > 1
    changeT = ndims(T) > 2
    changeR = ndims(R) > 2
    changeQ = ndims(Q) > 2
    changeA = ndims(a) > 1
    changeP = ndims(P) > 2
    
    ny, nobs = size(Y)
    ns = size(T,1)
    # QQ = R*Q*R'
    vR = view(R, :, :, 1)
    vQ = view(Q, :, :, 1)
    get_QQ!(ws.QQ, vR, vQ, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = 1
    steady = false
    vP = view(P, :, :, 1)
    copy!(ws.oldP, vP)
    @inbounds while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, pattern, t) : view(c, pattern)
        vZ = changeZ ? view(Z, :, :, t) : view(Z, :, :)
        vH = changeH ? view(H, :, :, t) : view(H, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vQ = changeR ? view(Q, :, :, t) : view(Q, :, :)
        va = changeR ? view(a, :, t) : view(a, :)
        va1 = changeA ? view(a, :, t + 1) : view(a, :)
        vd = changeD ? view(d, :, t) : view(d, :)
        vP = changeP ? view(P, :, :, t) : view(P, :, :)
        vP1 = changeP ? view(P, :, :, t + 1) : view(P, :, :)
        if changeR || changeQ
            get_QQ!(ws.QQ, vR, vQ, ws.RQ)
        end
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata, t)
        vK = view(ws.K, 1:ndata, :)
        
        # v  = Y[:,t] - c - Z*a
        get_v!(vv, Y, vc, vZ, va, t, pattern)
        iy += ny
        if !steady
            # F  = Z*P*Z' + H
            get_F!(vF, vZP, vZ, vP, vH)
            get_cholF!(vcholF, vF)
        end
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
        if t <= last
            if !steady
                # K = iF*ZP
                get_K!(vK, vZP, vcholF)
                # L = T(I - K'*Z)
                get_L!(vL, vT, vK, vZ) 
            end
            # a = d + T*a + K'*v
            update_a!(va1, va, vd, vK, vv, ws.a1, vT)
            if !steady
                # P = T*(P - K'*Z*P)*T'+ QQ
                copy!(vP1, vP)
                update_P!(vP1, vT, ws.QQ, vK, vZP, ws.PTmp)
                ws.oldP .-= vP
                if norm(ws.oldP) < ns*eps()
                    steady = true
                else
                    copy!(ws.oldP, vP)
                end
            end
        end
        t += 1
    end
    @inbounds if presample > 0
        LIK = -0.5*(sum(view(ws.lik,(presample+1):nobs)))
    else
        LIK = -0.5*(sum(ws.lik))
    end
    LIK
end

function kalman_smoother(Y::AbstractArray{U},
                         c::AbstractArray{U},
                         Z::AbstractArray{W},
                         H::AbstractArray{U},
                         d::AbstractArray{U},
                         T::AbstractArray{U},
                         R::AbstractArray{U},
                         Q::AbstractArray{U},
                         a::AbstractArray{U},
                         alphah::AbstractArray{U},
                         P::AbstractArray{U},
                         V::AbstractArray{U},
                         start::X,
                         last::X,
                         presample::X,
                         ws::KalmanSmootherWs,
                         data_pattern::Vector{Vector{X}}) where {U <: AbstractFloat, W <: Real, X <: Integer}

    kalman_filer_2(Y,c, Z, H, d, T, R, Q, a, alphah, P, V,start, last, presample, ws, data_pattern)

    fill!(ws.r, 0.0)
    fill!(ws.N, 0.0)

    for t = last: -1: 1
        pattern = data_pattern[t]
        ndata = length(pattern)
        vZ = changeZ ? view(Z, :, :, t) : view(Z, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        va = changeR ? view(a, :, t) : view(a, :)
        vP = changeP ? view(P, :, :, t) : view(P, :, :)
        viFv = view(ws.iFv, 1:ndata, t)
        valphah = view(alphah, :, t)
        
        # r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t
        update_r!(ws.r, vZ, viFv, vL, ws.r1)
        # alphah_t = a_t + P_t*r_{t-1}
        get_alphah!(valphah, va, P, ws.r)
        # N_{t-1} = Z_t'iF_t*Z_t + L_t'N_t*L_t
        update_N!(ws.N, vZ, viFZ, vL, ws.N1, ws.Ptmp)
        # V_t = P_t - P_t*N_{t-1}*P_t
        get_V!(vV, vP, ws.N, ws.Ptmp)
    end
end

end #module