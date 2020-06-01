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

include("kalman_base.jl")
include("univariate_step.jl")

"""
State space specification:
    y_t = Z*a_t + epsilon_t
    a_{t+1} = T*a_t- + R eta_t
    E(epsilon_t epsilon_t') = H
    E(eta_t eta_t') = Q
"""

using LinearAlgebra
using LinearAlgebra.BLAS

export KalmanLikelihoodWs, FastKalmanLikelihoodWs, DiffuseKalmanLikelihoodWs, KalmanSmootherWs, kalman_likelihood, kalman_likelihood_monitored, fast_kalman_likelihood, diffuse_kalman_likelihood, kalman_filter!, kalman_smoother!

abstract type KalmanWs{T, U} end

struct KalmanLikelihoodWs{T, U} <: KalmanWs{T, U}
    csmall::Vector{T}
    Zsmall::Matrix{T}
    # necessary for Z selecting vector with missing variables
    iZsmall::Vector{U}
    RQ::Matrix{T}
    QQ::Matrix{T}
    v::Vector{T}
    F::Matrix{T}
    cholF::Matrix{T}
    cholH::Matrix{T}
    LTcholH::Matrix{T}
    iFv::Vector{T}
    a1::Vector{T}
    K::Matrix{T}
    ZP::Matrix{T}
    iFZ::SubArray{T}
    PTmp::Matrix{T}
    oldP::Matrix{T}
    lik::Vector{T}
    cholHset::Bool
    ystar::Vector{T}
    Zstar::Matrix{T}
    tmp_ns::Vector{T}
    PZi::Vector{T}
    kalman_tol::T

    function KalmanLikelihoodWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        csmall = Vector{T}(undef, ny)
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        RQ = Matrix{T}(undef, ns, np)
        QQ = Matrix{T}(undef, ns, ns)
        F = Matrix{T}(undef, ny, ny)
        cholF = Matrix{T}(undef, ny, ny)
        cholH = Matrix{T}(undef, ny, ny)
        LTcholH = Matrix{T}(undef, ny, ny)
        v = Vector{T}(undef, ny)
        iFv = Vector{T}(undef, ny)
        a1 = Vector{T}(undef, ns)
        K = Matrix{T}(undef, ny, ns)
        PTmp = Matrix{T}(undef, ns, ns)
        oldP = Matrix{T}(undef, ns, ns)
        ZP = Matrix{T}(undef, ny, ns)
        iFZ = view(PTmp,1:ny,:)
        lik = Vector{T}(undef, nobs)
        cholHset = false
        ystar = Vector{T}(undef, ny)
        Zstar = Matrix{T}(undef, ny, ns)
        tmp_ns = Vector{T}(undef, ns)
        PZi = Vector{T}(undef, ns)
        kalman_tol = 1e-12

         new(csmall, Zsmall, iZsmall, RQ, QQ, v, F, cholF, cholH, LTcholH,
            iFv, a1, K, ZP, iFZ, PTmp, oldP, lik, cholHset, ystar,
            Zstar, tmp_ns, PZi, kalman_tol)
    end
end

KalmanLikelihoodWs(ny, ns, np, nobs) = KalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

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
    ny = size(Y,1)
    nobs = last - start + 1
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    fill!(ws.lik, 0.0)
    cholHset = false
    t = start
    iy = (start - 1)*ny + 1
    @inbounds while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        # F  = Z*P*Z' + H
        get_F!(ws.F, ws.ZP, Z, P, H)
        info = get_cholF!(ws.cholF, ws.F)
        if info != 0
            # F is near singular
            if !cholHset
                get_cholF!(ws.cholH, H)
                cholHset = true
            end
            ws.lik[t] = univariate_step!(Y, t, Z, H, T, ws.QQ, a, P, ws.kalman_tol, ws)
        else
            # iFv = inv(F)*v
            get_iFv!(ws.iFv, ws.cholF, ws.v)
            ws.lik[t] = log(det_from_cholesky(ws.cholF)) + LinearAlgebra.dot(ws.v, ws.iFv)
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
    ny = size(Y,1)
    nobs = last - start + 1
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = (start - 1)*ny + 1
    ncolZ = size(Z, 2)
    cholHset = false
    @inbounds while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vH = view(H, pattern, pattern)
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        vcholH = view(ws.cholF, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata)
        vK = view(ws.K, 1:ndata, :)
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, Z, pattern, ndata, ny)

        # v  = Y[:,t] - Z*a
        get_v!(vv, Y, vZsmall, a, t, pattern)
        iy += ny
        # F  = Z*P*Z' + H
        get_F!(vF, vZP, vZsmall, P, vH)
        info = get_cholF!(vcholF, vF)
        if info != 0
            @show info
            # F is near singular
            if !cholHset
                get_cholF!(vcholH, vH)
                cholHset = true
            end
            ws.lik[t] = univariate_step!(Y, t, Z, H, T, ws.QQ, a, P, ws.kalman_tol, ws)
        else
            # iFv = inv(F)*v
            get_iFv!(viFv, vcholF, vv)
            ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
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
    ny = size(Y,1)
    nobs = last - start + 1
    ns = size(T,1)
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = (start - 1)*ny + 1
    steady = false
    copy!(ws.oldP, P)
    @inbounds while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        if !steady
            # F  = Z*P*Z' + H
            get_F!(ws.F, ws.ZP, Z, P, H)
            info = get_cholF!(ws.cholF, ws.F)
            if info != 0
                @show info
                # F is near singular
                if !cholHset
                    get_cholF!(ws.cholH, H)
                    cholHset = true
                end
                ws.lik[t] = univariate_step!(Y, t, Z, H, T, ws.QQ, a, P, ws.kalman_tol, ws)
                t += 1
                continue
            end
        end
        # iFv = inv(F)*v
        get_iFv!(ws.iFv, ws.cholF, ws.v)
        ws.lik[t] = log(det_from_cholesky(ws.cholF)) + LinearAlgebra.dot(ws.v, ws.iFv)
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
    ny = size(Y,1)
    nobs = last - start + 1
    ns = size(T,1)
    # QQ = R*Q*R'
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = (start - 1)*ny + 1
    steady = false
    copy!(ws.oldP, P)
    cholHset = false
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
            info = get_cholF!(ws.cholF, ws.F)
            if info != 0
                # F is near singular
                if !cholHset
                    get_cholF!(ws.cholH, H)
                    cholHset = true
                end
                ws.lik[t] = univariate_step!(Y, t, Z, H, T, ws.QQ, a, P, ws.kalman_tol, ws)
                t += 1
                continue
            end
        end
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
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
    csmall::Vector{T}
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
    kalman_tol::T

    function FastKalmanLikelihoodWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        csmall = Vector{T}(undef, ny)
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
        kalman_tol = 1e-12
        new(csmall, Zsmall, iZsmall, QQ, v, F, cholF, iFv, a1, K, RQ, ZP, M, W, ZW, ZWM, iFZWM, TW, iFZW, KtiFZW, lik, kalman_tol)
    end
end

FastKalmanLikelihoodWs(ny, ns, np, nobs) = FastKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

"""
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
    ny = size(Y,1)
    nobs = last - start + 1
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
    iy = (start - 1)*ny + 1
    @inbounds while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        # iFv = inv(F)*v
        get_iFv!(ws.iFv, ws.cholF, ws.v)
        ws.lik[t] = log(det_from_cholesky(ws.cholF)) + LinearAlgebra.dot(ws.v, ws.iFv)
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
    ny = size(Y,1)
    nobs = last - start + 1
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
    iy = (start - 1)*ny + 1
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
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, Z, pattern, ndata, ny)

        # v  = Y[:,t] - Z*a
        get_v!(vv, Y, vZsmall, a, t, pattern)
        iy += ny
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
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
    csmall::Vector{T}
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
    kalman_tol::T
    function DiffuseKalmanLikelihoodWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        csmall = Vector{T}(undef, ny)
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
        kalman_tol = 1e-12
        new(csmall, Zsmall, iZsmall, QQ, RQ, v, F, iF, iFv, a1, cholF, ZP, Fstar, ZPstar, K, iFZ, Kstar, PTmp, uKinf, uKstar, Kinf_Finf, lik, kalman_tol)
    end
end

DiffuseKalmanLikelihoodWs(ny, ns, np, nobs) = DiffuseKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

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
    iy = (start - 1)*ny + 1
    diffuse_kalman_tol = 1e-8
    kalman_tol = 1e-8
    while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        # Finf = Z*Pinf*Z'
        get_F!(ws.F, ws.ZP, Z, Pinf)
        info = get_cholF!(ws.cholF, ws.F)
        if info > 0
            if norm(ws.F) < tol
                return t - 1
            else
                ws.lik[t] += univariate_step(Y, t, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws)
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
    iy = (start - 1)*ny + 1
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
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, Z, pattern, ndata, ny)

        # v  = Y[:,t] - Z*a
        get_v!(vv, Y, vZsmall, a, t, pattern)
        iy += ny
        # Finf = Z*Pinf*Z'
        get_F!(vF, vZP, vZsmall, Pinf)
        info = get_cholF!(vcholF, vF)
        if info > 0
            if norm(vF) < tol
                return t - 1
            else
                ws.lik[t] += univariate_step(Y, t, vZsmall, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, pattern, ws)
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
    ny = size(Y,1)
    nobs = last - start + 1
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
    ny = size(Y,1)
    nobs = last - start + 1
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

# Filters

function kalman_filter!(Y::AbstractArray{U},
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

    ny = size(Y, 1)
    nobs = last - start + 1
    ns = size(T,1)
    # QQ = R*Q*R'
    vR = view(R, :, :, 1)
    vQ = view(Q, :, :, 1)
    get_QQ!(ws.QQ, vR, vQ, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    iy = (start - 1)*ny + 1
    steady = false
    vP = view(P, :, :, 1)
    copy!(ws.oldP, vP)
    cholHset = false
    @inbounds while t <= last

        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, :, t) : view(c, :)
        ws.csmall .= view(vc, pattern)
        vZ = changeZ ? view(Z, :, :, t) : view(Z, :, :)
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, vZ, pattern, ndata, ny)
        vH = changeH ? view(H, :, :, t) : view(H, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
        va = changeA ? view(a, :, t) : view(a, :)
        va1 = changeA ? view(a, :, t + 1) : view(a, :)
        vd = changeD ? view(d, :, t) : view(d, :)
        vP = changeP ? view(P, :, :, t) : view(P, :, :)
        vP1 = changeP ? view(P, :, :, t + 1) : view(P, :, :)
        if changeR || changeQ
            get_QQ!(ws.QQ, vR, vQ, ws.RQ)
        end
        vv = view(ws.v, 1:ndata)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vvH = view(vH, pattern, pattern)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata, 1)
        viFv = view(ws.iFv, 1:ndata)
        vK = view(ws.K, 1:ndata, :, 1)

        # v  = Y[:,t] - c - Z*a
        get_v!(vv, Y, vc, vZsmall, va, t, pattern)
        iy += ny
        if !steady
            # F  = Z*P*Z' + H
            get_F!(vF, vZP, vZsmall, vP, vvH)
            info = get_cholF!(vcholF, vF)
            if info != 0
                # F is near singular
                if !cholHset
                    get_cholF!(ws.cholH, vvH)
                    cholHset = true
                end
                ws.lik[t] = univariate_step!(Y, t, ws.Zsmall, vvH, T, ws.QQ, va, vP, ws.kalman_tol, ws)
                t += 1
                continue
            end
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
                ws.oldP .-= vP1
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
    csmall::Vector{T}
    Zsmall::Matrix{T}
    # necessary for Z selecting vector with missing variables
    iZsmall::Vector{U}
    RQ::Matrix{T}
    QQ::Matrix{T}
    v::Matrix{T}
    F::Matrix{T}
    cholF::Matrix{T}
    iF::Array{T}
    iFv::Matrix{T}
    r::Vector{T}
    r1::Vector{T}
    at_t::Matrix{T}
    K::Matrix{T}
    KDK::Matrix{T}
    L::Matrix{T}
    L1::Matrix{T}
    N::Matrix{T}
    N1::Matrix{T}
    ZP::Matrix{T}
    Pt_t::Matrix{T}
    Kv::Matrix{T}
    iFZ::Matrix{T}
    PTmp::Matrix{T}
    oldP::Matrix{T}
    lik::Vector{T}
    KT::Matrix{T}
    D::Matrix{T}
    tmp_np::Vector{T}
    tmp_ns::Vector{T}
    tmp_ny::Vector{T}
    tmp_ns_np::AbstractArray{T}
    tmp_ny_ny::AbstractArray{T}

    function KalmanSmootherWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        csmall = Vector{T}(undef, ny)
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        RQ = Matrix{T}(undef, ns, np)
        QQ = Matrix{T}(undef, ns, ns)
        F = Matrix{T}(undef, ny, ny)
        cholF = Matrix{T}(undef, ny, ny)
        iF = Array{T}(undef, ny, ny, nobs)
        v = Matrix{T}(undef, ny, nobs)
        iFv = Matrix{T}(undef, ny, nobs)
        r = zeros(T, ns)
        r1 = zeros(T, ns)
        at_t = zeros(T, ns, nobs)
        K = Array{T}(undef, ny, ns, nobs)
        KDK = Array{T}(undef, ny, ns, nobs)
        L = Matrix{T}(undef, ns, ns)
        L1 = Matrix{T}(undef, ns, ns)
        N = zeros(T, ns, ns)
        N1 = zeros(T, ns, ns)
        Kv = Matrix{T}(undef, ns, nobs)
        Pt_t = zeros(T, ns, ns, nobs)
        PTmp = Matrix{T}(undef, ns, ns)
        oldP = Matrix{T}(undef, ns, ns)
        ZP = Matrix{T}(undef, ny, ns)
        iFZ = Matrix{T}(undef, ny, ns)
        lik = Vector{T}(undef, nobs)
        KT = Matrix{T}(undef, ny, ns)
        D = Matrix{T}(undef, ny, ny)
        tmp_np = Vector{T}(undef, np)
        tmp_ns = Vector{T}(undef, ns)
        tmp_ny = Vector{T}(undef, ny)
        tmp_ns_np = Matrix{T}(undef, ns, np)
        tmp_ny_ny = Matrix{T}(undef, ny, ny)

        new(csmall, Zsmall, iZsmall, RQ, QQ, v, F, cholF, iF,
            iFv, r, r1, at_t, K, KDK, L, L1, N, N1, ZP, Pt_t, Kv,
            iFZ, PTmp, oldP, lik, KT, D, tmp_np, tmp_ns,
            tmp_ny, tmp_ns_np, tmp_ny_ny)
    end
end

KalmanSmootherWs(ny, ns, np, nobs) = KalmanSmootherWs{Float64, Int64}(ny, ns, np, nobs)

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

    ny = size(Y,1)
    nobs = last - start + 1
    ns = size(T,1)
    # QQ = R*Q*R'
    vR = view(R, :, :, 1)
    vQ = view(Q, :, :, 1)
    va = view(a,:)
    vP = view(P, :, :)
    get_QQ!(ws.QQ, vR, vQ, ws.RQ)
    l2pi = log(2*pi)
    fill!(ws.lik, 0.0)
    t = start
    steady = false
#    copy!(ws.oldP, vP)
    while t <= last
        #inputs
        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, pattern, t) : view(c, pattern)
        vZ = changeZ ? view(Z, :, :, t) : view(Z, :, :)
        vH = changeH ? view(H, :, :, t) : view(H, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        vd = changeD ? view(d, :, t) : view(d, :)
        if changeR || changeQ
            vR = view(R, :, :, t)
            vQ = view(Q, :, :, t)
            get_QQ!(ws.QQ, vR, vQ, ws.RQ)
        end
        # outputs
        vat = view(ws.at_t, :, t)
        vPt = view(ws.Pt_t, :, :, t)
        vv = view(ws.v, 1:ndata, t)
        vF = view(ws.F, 1:ndata, 1:ndata)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata)
        viF = view(ws.iF, 1:ndata, 1:ndata, t)
        viFv = view(ws.iFv, 1:ndata, t)
        vK = view(ws.K, 1:ndata, :, t)
        vKDK = view(ws.KDK, 1:ndata, :, t) # Kalman Filter Gain according to Durbin & Koopman (4.22)

        # v  = Y[:,t] - c - Z*a
        get_v!(vv, Y, vc, vZ, va, t, pattern)
#        iy += ny
#        if !steady
            # F  = Z*P*Z' + H
            # builds also ZP
        get_F!(vF, vZP, vZ, vP, vH)
            info = get_cholF!(vcholF, vF)
            if info != 0
                # F is near singular
                if !cholHset
                    get_cholF!(ws.cholH, H)
                    cholHset = true
                end
                ws.lik[t] = univariate_step!(Y, t, vZ, vH, vT, ws.QQ, a, P, ws.kalman_tol, ws)
                t += 1
                continue
            end
#        end
        # iFv = inv(F)*v
        get_iF!(viF, vcholF)
        mul!(viFv,viF,vv)
#        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = -.5*ndata*l2pi -.5*log(det_from_cholesky(vcholF)) -.5*LinearAlgebra.dot(vv, viFv)
#        if t <= last
#            if !steady
                # K = iF*ZP
            mul!(vK,viF,vZP)
                # amounts to K_t in DK (4.22): here KDK = T*K'
            mul!(vKDK,vT,transpose(vK))
#            end
            # a{t_t} = d + a_t + K'*v
            filtered_a!(vat, va, vd, vK, vv, ws.tmp_ns)
            # a_{t+1} = T a_{t_t}
            mul!(va,vT,vat)
#            if !steady
                # P_{t|t} = P_t - K'*Z*P_t
            filtered_P!(vPt, vP, vK, vZP, ws.PTmp)
                # P_{t+1} = T*P_{t|t}*T'+ QQ
            update_P!(vP, vPt, vT, ws.QQ, ws.PTmp)

#=
                ws.oldP .-= vP1
                if norm(ws.oldP) < ns*eps()
                    steady = true
                else
                    copy!(ws.oldP, vP)
                end
=#
#            end
#        end
        t += 1
    end
end

function kalman_smoother!(Y::AbstractArray{U},
                          c::AbstractArray{U},
                          Z::AbstractArray{W},
                          H::AbstractArray{U},
                          d::AbstractArray{U},
                          T::AbstractArray{U},
                          R::AbstractArray{U},
                          Q::AbstractArray{U},
                          a::AbstractArray{U},
                          P::AbstractArray{U},
                          alphah::AbstractArray{U},
                          epsilonh::AbstractArray{U},
                          etah::AbstractArray{U},
                          Valpha::AbstractArray{U},
                          Vepsilon::AbstractArray{U},
                          Veta::AbstractArray{U},
                          start::X,
                          last::X,
                          presample::X,
                          ws::KalmanSmootherWs,
                          data_pattern::Vector{Vector{X}}) where {U <: AbstractFloat, W <: Real, X <: Integer}

    changeC = ndims(c) > 1
    changeZ = ndims(Z) > 2
    changeH = ndims(H) > 2
    changeD = ndims(d) > 1
    changeT = ndims(T) > 2
    changeR = ndims(R) > 2
    changeQ = ndims(Q) > 2
    changeA = ndims(a) > 1
    changeP = ndims(P) > 2

    kalman_filter_2!(Y,c, Z, H, d, T, R, Q, a,
                     P, start, last, presample, ws,
                     data_pattern)

    fill!(ws.r1,0.0)
    fill!(ws.N1,0.0)

    for t = last: -1: 1
        #inputs
        pattern = data_pattern[t]
        ndata = length(pattern)
        vZ = changeZ ? view(Z, :, :, t) : view(Z, :, :)
        vH = changeH ? view(H, :, :, t) : view(H, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        va = view(ws.at_t, :, t)
        vP = view(ws.Pt_t, :, :, t)
        vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vKDK = view(ws.KDK, 1:ndata, :, t) # amounts to K_t (4.22): here KDK = T*K'
        viF = view(ws.iF, 1:ndata, 1:ndata, t)
        viFv = view(ws.iFv, 1:ndata, t)

        # L_t = T - KDK_t*Z (DK 4.29)
        get_L!(ws.L, vT, vKDK, vZ, ws.L1)
        # r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t (DK 4.44)
        update_r!(ws.r, vZ, viFv, ws.L, ws.r1)
        if (length(alphah) > 0 ||
            length(epsilonh) > 0 ||
            length(etah) > 0)
            # N_{t-1} = Z_t'iF_t*Z_t + L_t'N_t*L_t (DK 4.44)
            mul!(ws.iFZ,viF,vZ)
            update_N!(ws.N, vZ, ws.iFZ, ws.L, ws.N1, ws.PTmp)
        end
        if length(epsilonh) > 0
            vepsilonh = view(epsilonh, :, t)
            # epsilon_t = H*(iF_t*v_t - KDK_t'*r_t) (DK 4.69)
            get_epsilonh!(vepsilonh, vH, viFv, vKDK, ws.r1, ws.tmp_ny, ws.tmp_ns)
            if length(Vepsilon) > 0
                vVepsilon = view(Vepsilon,:,:,t)
                # D_t = inv(F_t) + KDK_t'*N_t*KDK_t (DK 4.69)
                get_D!(ws.D, viF, vKDK,  ws.N1, ws.tmp_ns_np)
                # Vepsilon_t = H - H*D_t*H (DK 4.69)
                get_Vepsilon!(vVepsilon, vH, ws.D, ws.tmp_ny_ny)
            end
        end
        if length(etah) > 0
            vetah = view(etah, :, t)
            # eta_t = Q*R'*r_t (DK 4.69)
            get_etah!(vetah, vQ, vR, ws.r1, ws.tmp_np)
            if length(Veta) > 0
                vVeta = view(Veta, :, :, t)
                # Veta_t = Q - Q*R'*N_t*R*Q (DK 4.69)
                get_Veta!(vVeta, vQ, vR, ws.N1, ws.RQ, ws.tmp_ns_np)
            end
        end

        if length(alphah) > 0
            valphah = view(alphah, :, t)
            if t==last
                valphah .= va
            else
                # alphah_t = a_t + P_t*r_{t-1} (DK 4.44)
                get_alphah!(valphah, va, vP, ws.r1)
            end
        end
        if length(Valpha) > 0
            vValpha = view(Valpha, :, :, t)
            if t==last
                copy!(vValpha,vP)
            else
                # Valpha_t = P_t - P_t*N_{t-1}*P_t (DK 4.44)
                get_Valpha!(vValpha, vP, ws.N1, ws.PTmp)
            end
        end
#=
        if length(alphah) > 0
            valphah = view(alphah, :, t)
            # alphah_t = a_t + P_t*r_{t-1} (DK 4.44)
            get_alphah!(valphah, va, vP, ws.r)
        end
        if length(Valpha) > 0
            vValpha = view(Valpha, :, :, t)
            # Valpha_t = P_t - P_t*N_{t-1}*P_t (DK 4.44)
            get_Valpha!(vValpha, vP, ws.N, ws.PTmp)
        end
=#
        copy!(ws.r1,ws.r)
        copy!(ws.N1,ws.N)
    end
end

end #module
