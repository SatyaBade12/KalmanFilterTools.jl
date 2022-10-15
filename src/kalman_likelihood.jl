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
    Hstar::Matrix{T}
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
        Hstar = Matrix{T}(undef, ny, ny)
        tmp_ns = Vector{T}(undef, ns)
        PZi = Vector{T}(undef, ns)
        kalman_tol = 1e-12

         new(csmall, Zsmall, iZsmall, RQ, QQ, v, F, cholF, cholH, LTcholH,
            iFv, a1, K, ZP, iFZ, PTmp, oldP, lik, cholHset, ystar,
            Zstar, Hstar, tmp_ns, PZi, kalman_tol)
    end
end

KalmanLikelihoodWs(ny, ns, np, nobs) = KalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

# Z can be either a matrix or a selection vector
function kalman_likelihood(Y::AbstractMatrix{X},
                           Z::AbstractVecOrMat{W},
                           H::AbstractMatrix{U},
                           T::AbstractMatrix{U},
                           R::AbstractMatrix{U},
                           Q::AbstractMatrix{U},
                           a::AbstractVector{U},
                           P::AbstractMatrix{U},
                           start::V,
                           last::V,
                           presample::V,
                           ws::KalmanWs) where {U <: AbstractFloat, W <: Real, V <: Integer, X <: Union{AbstractFloat, Missing}}
    ny = size(Y,1)
    nobs = last - start + 1
    # QQ = R*Q*R'
    @inbounds get_QQ!(ws.QQ, R, Q, ws.RQ)
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
    lik_cst = (nobs - presample)*ny*log(2*pi)
    @inbounds vlik = view(ws.lik, start + presample:last)
    return @inbounds -0.5*(lik_cst + sum(vlik))
end

function kalman_likelihood(Y::AbstractMatrix{X},
                           Z::AbstractVecOrMat{W},
                           H::AbstractMatrix{U},
                           T::AbstractMatrix{U},
                           R::AbstractMatrix{U},
                           Q::AbstractMatrix{U},
                           a::AbstractVector{U},
                           P::AbstractMatrix{U},
                           start::V,
                           last::V,
                           presample::V,
                           ws::KalmanWs,
                           data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat, W <: Real, V <: Integer, X <: Union{AbstractFloat, Missing}}
    ny = size(Y,1)
    nobs = last - start + 1
    # QQ = R*Q*R'
    @inbounds get_QQ!(ws.QQ, R, Q, ws.RQ)
    l2pi = log(2*pi)
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
            # F is near singular
            if !cholHset
                get_cholF!(vcholH, vH)
                cholHset = true
            end
            ws.lik[t] = ndata*l2pi + univariate_step!(Y, t, Z, H, T, ws.QQ, a, P, ws.kalman_tol, ws, pattern)
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
    @inbounds vlik = view(ws.lik, start + presample:last)
    return @inbounds -0.5*sum(vlik)
end

function kalman_likelihood_monitored(Y::AbstractMatrix{X},
                                     Z::AbstractVecOrMat{W},
                                     H::AbstractMatrix{U},
                                     T::AbstractMatrix{U},
                                     R::AbstractMatrix{U},
                                     Q::AbstractMatrix{U},
                                     a::Vector{U},
                                     P::AbstractMatrix{U},
                                     start::V,
                                     last::V,
                                     presample::V,
                                     ws::KalmanWs) where {U <: AbstractFloat, V <: Integer, W <: Real, X <: Union{AbstractFloat, Missing}}
    ny = size(Y,1)
    nobs = last - start + 1
    ns = size(T,1)
    # QQ = R*Q*R'
    @inbounds get_QQ!(ws.QQ, R, Q, ws.RQ)
    t = start
    iy = (start - 1)*ny + 1
    steady = false
    @inbounds copy!(ws.oldP, P)
    cholHset = false
    @inbounds while t <= last
        # v  = Y[:,t] - Z*a
        get_v!(ws.v, Y, Z, a, iy, ny)
        iy += ny
        if !steady
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
            if norm(ws.oldP) < 0*ns*eps()
                steady = true
            else
                copy!(ws.oldP, P)
            end
        end
        t += 1
    end
    lik_cst = (nobs - presample)*ny*log(2*pi)
    @inbounds vlik = view(ws.lik, start + presample:last)
    return @inbounds -0.5*(lik_cst + sum(vlik))
end

function kalman_likelihood_monitored(Y::AbstractMatrix{X},
                                     Z::AbstractVecOrMat{W},
                                     H::AbstractMatrix{U},
                                     T::AbstractMatrix{U},
                                     R::AbstractMatrix{U},
                                     Q::AbstractMatrix{U},
                                     a::AbstractVector{U},
                                     P::AbstractMatrix{U},
                                     start::V,
                                     last::V,
                                     presample::V,
                                     ws::KalmanWs,
                                     data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat, W <: Real, V <: Integer, X <: Union{AbstractFloat, Missing}}
    ny = size(Y,1)
    nobs = last - start + 1
    ns = size(T,1)
    # QQ = R*Q*R'
    @inbounds get_QQ!(ws.QQ, R, Q, ws.RQ)
    l2pi = log(2*pi)
    t = start
    iy = (start - 1)*ny + 1
    steady = false
    @inbounds copy!(ws.oldP, P)
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
            info = get_cholF!(vcholF, ws.F)
            if info != 0
                # F is near singular
                if !cholHset
                    get_cholF!(ws.cholH, H)
                    cholHset = true
                end
                ws.lik[t] = ndata*l2pi + univariate_step!(Y, t, Z, H, T, ws.QQ, a, P, ws.kalman_tol, ws, pattern)
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
            if norm(ws.oldP) < 0*ns*eps()
                steady = true
            else
                copy!(ws.oldP, P)
            end
        end
        t += 1
    end
    @inbounds vlik = view(ws.lik, start + presample:last)
    return @inbounds -0.5*sum(vlik)
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
    ystar::Vector{T}
    Zstar::Matrix{T}
    Hstar::Matrix{T}
    PZi::Vector{T}
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
        ystar = Vector{T}(undef, ny)
        Zstar = Matrix{T}(undef, ny, ns)
        Hstar = Matrix{T}(undef, ny, ny)
        PZi = Vector{T}(undef, ns)
        lik = Vector{T}(undef, nobs)
        kalman_tol = 1e-12
        new(csmall, Zsmall, iZsmall, QQ, v, F, cholF, iFv, a1, K, RQ, ZP, M, W,
            ZW, ZWM, iFZWM, TW, iFZW, KtiFZW, ystar, Zstar, Hstar, PZi, lik, kalman_tol)
    end
end

FastKalmanLikelihoodWs(ny, ns, np, nobs) = FastKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

"""
K doesn't represent the same matrix as above
"""
function fast_kalman_likelihood(Y::Matrix{U},
                                Z::AbstractVecOrMat{W},
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
    # F  = Z*P*Z' + H
    get_F!(ws.F, ws.ZP, Z, P, H)
    get_cholF!(ws.cholF, ws.F)
    # K = Z*P
    copy!(ws.K, ws.ZP)
    # W = T*K'
    mul!(ws.W, T, transpose(ws.K))
    # M = -iF
    get_M!(ws.M, ws.cholF, ws.ZW)
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
    lik_cst = (nobs - presample)*ny*log(2*pi)
    vlik = view(ws.lik, start + presample:last)
    return -0.5*(lik_cst + sum(vlik))
end

function fast_kalman_likelihood(Y::Matrix{U},
                                Z::AbstractVecOrMat{W},
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
    # F  = Z*P*Z' + H
    get_F!(ws.F, ws.ZP, Z, P, H)
    get_cholF!(ws.cholF, ws.F)
    # K = Z*P
    copy!(ws.K, ws.ZP)
    # W = T*K'
    mul!(ws.W, T, transpose(ws.K))
    # M = -iF
    get_M!(ws.M, ws.cholF, ws.ZW)
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
    vlik = view(ws.lik, start + presample:last)
    return -0.5*sum(vlik)
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
    ystar::Vector{T}
    Zstar::Matrix{T}
    Hstar::Matrix{T}
    PZi::Vector{T}
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
        ystar = Vector{T}(undef, ny)
        Zstar = Matrix{T}(undef, ny, ns)
        Hstar = Matrix{T}(undef, ny, ny)
        PZi = Vector{T}(undef, ns)
        lik = zeros(T, nobs)
        kalman_tol = 1e-12
        new(csmall, Zsmall, iZsmall, QQ, RQ, v, F, iF, iFv, a1, cholF, ZP, Fstar,
            ZPstar, K, iFZ, Kstar, PTmp, uKinf, uKstar, Kinf_Finf, ystar, Zstar, Hstar,
            PZi, lik, kalman_tol)
    end
end

DiffuseKalmanLikelihoodWs(ny, ns, np, nobs) = DiffuseKalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

function diffuse_kalman_likelihood_init!(Y::Matrix{U},
                                         Z::AbstractVecOrMat{W},
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
                                         Z::AbstractVecOrMat{W},
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
    iy = (start - 1)*ny + 1
    diffuse_kalman_tol = 1e-8
    l2pi = log(2*pi)
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
                ws.lik[t] += ndata*l2pi + univariate_step(Y, t, vZsmall, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, pattern, ws, pattern)
            end
        else
            ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF))
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
                                   Z::AbstractVecOrMat{W},
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
    t = diffuse_kalman_likelihood_init!(Y, Z, H, T, ws.QQ, a, Pinf, Pstar, start, last, tol, ws)
    kalman_likelihood(Y, Z, H, T, R, Q, a, Pstar, t + 1, last, presample, ws)
    lik_cst = (nobs - presample)*ny*log(2*pi)
    vlik = view(ws.lik, start + presample:last)
    return -0.5*(lik_cst + sum(vlik))
end

function diffuse_kalman_likelihood(Y::Matrix{U},
                                   Z::AbstractVecOrMat{W},
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
    t = diffuse_kalman_likelihood_init!(Y, Z, H, T, ws.QQ, a, Pinf, Pstar, start, last, tol, ws, data_pattern)
    kalman_likelihood(Y, Z, H, T, R, Q, a, Pstar, t + 1, last, presample, ws, data_pattern)
    vlik = view(ws.lik, start + presample:last)
    return -0.5*sum(vlik)
end

