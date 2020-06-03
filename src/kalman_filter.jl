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

struct DiffuseKalmanFilterWs{T, U} <: KalmanWs{T, U}
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

    function DiffuseKalmanFilterWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
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
            PZI, lik, kalman_tol)
    end
end

DiffuseKalmanFilterWs(ny, ns, np, nobs) = DiffuseKalmanFilterWs{Float64, Int64}(ny, ns, np, nobs)

function diffuse_kalman_filter_init!(Y::AbstractArray{U},
                                     c::AbstractArray{U},
                                     Z::AbstractArray{W},
                                     H::AbstractArray{U},
                                     d::AbstractArray{U},
                                     T::AbstractArray{U},
                                     QQ::AbstractArray{U},
                                     a::AbstractArray{U},
                                     Pinf::AbstractArray{U},
                                     Pstar::AbstractArray{U},
                                     start::V,
                                     last::V,
                                     presample::V,
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

function diffuse_kalman_filter_init!(Y::Matrix{U},
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

function diffuse_kalman_filter(Y::Matrix{U},
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

function diffuse_kalman_filter(Y::Matrix{U},
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

