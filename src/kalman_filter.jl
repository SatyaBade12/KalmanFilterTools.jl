# Filters
struct KalmanFilterWs{T, U} <: KalmanWs{T, U}
    csmall::Vector{T}
    Zsmall::Matrix{T}
    # necessary for Z selecting vector with missing variables
    iZsmall::Vector{U}
    RQ::Matrix{T}
    QQ::Matrix{T}
    v::Matrix{T}
    F::Matrix{T}
    cholF::Array{T}
    cholH::Matrix{T}
    iF::Array{T}
    iFv::Array{T}
    a1::Vector{T}
    r::Vector{T}
    r1::Vector{T}
    at_t::Matrix{T}
    K::Array{T}
    KDK::Array{T}
    L::Matrix{T}
    L1::Matrix{T}
    N::Matrix{T}
    N1::Matrix{T}
    ZP::Matrix{T}
    Kv::Matrix{T}
    iFZ::Matrix{T}
    PTmp::Matrix{T}
    oldP::Matrix{T}
    lik::Vector{T}
    KT::Matrix{T}
    D::Matrix{T}
    ystar::Vector{T}
    Zstar::Matrix{T}
    Hstar::Matrix{T}
    PZi::Vector{T}
    tmp_np::Vector{T}
    tmp_ns::Vector{T}
    tmp_ny::Vector{T}
    tmp_ns_np::AbstractArray{T}
    tmp_ny_ny::AbstractArray{T}
    tmp_ny_ns::AbstractArray{T}
    kalman_tol::T
    
    function KalmanFilterWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        csmall = Vector{T}(undef, ny)
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        RQ = Matrix{T}(undef, ns, np)
        QQ = Matrix{T}(undef, ns, ns)
        F = Matrix{T}(undef, ny, ny)
        cholF = Array{T}(undef, ny, ny, nobs)
        cholH = Matrix{T}(undef, ny, ny)
        iF = Array{T}(undef, ny, ny, nobs)
        a1 = Vector{T}(undef, ns)
        v = Matrix{T}(undef, ny, nobs)
        iFv = Array{T}(undef, ny)
        r = zeros(T, ns)
        r1 = zeros(T, ns)
        at_t = zeros(T, ns, nobs)
        K = Array{T}(undef, ny, ns, nobs)
        KDK = Array{T}(undef, ns, ny, nobs)
        L = Matrix{T}(undef, ns, ns)
        L1 = Matrix{T}(undef, ns, ns)
        N = zeros(T, ns, ns)
        N1 = zeros(T, ns, ns)
        Kv = Matrix{T}(undef, ns, nobs)
        PTmp = Matrix{T}(undef, ns, ns)
        oldP = Matrix{T}(undef, ns, ns)
        ZP = Matrix{T}(undef, ny, ns)
        iFZ = Matrix{T}(undef, ny, ns)
        lik = Vector{T}(undef, nobs)
        KT = Matrix{T}(undef, ny, ns)
        D = Matrix{T}(undef, ny, ny)
        ystar = Vector{T}(undef, ny)
        Zstar = Matrix{T}(undef, ny, ns)
        Hstar = Matrix{T}(undef, ny, ny)
        PZi = Vector{T}(undef, ns)
        tmp_np = Vector{T}(undef, np)
        tmp_ns = Vector{T}(undef, ns)
        tmp_ny = Vector{T}(undef, ny)
        tmp_ns_np = Matrix{T}(undef, ns, np)
        tmp_ny_ny = Matrix{T}(undef, ny, ny)
        tmp_ny_ns = Matrix{T}(undef, ny, ns)
        kalman_tol = 1e-12

        new(csmall, Zsmall, iZsmall, RQ, QQ, v, F, cholF, cholH, iF,
            iFv, a1, r, r1, at_t, K, KDK, L, L1, N, N1, ZP, Kv,
            iFZ, PTmp, oldP, lik, KT, D, ystar, Zstar, Hstar, PZi,
            tmp_np, tmp_ns, tmp_ny, tmp_ns_np, tmp_ny_ny, tmp_ny_ns,
            kalman_tol)
    end
end

KalmanFilterWs(ny, ns, np, nobs) = KalmanFilterWs{Float64, Int64}(ny, ns, np, nobs)

function simple_update!(va1, va, vd, vP1, vP, vT, steady, ws)
    # a = d + T*a
    update_a!(va1, vd, vT, va)
    if !steady
        copy!(ws.oldP, vP)
        # P = T*P*T + QQ
        update_P!(vP1, vT, vP, ws.QQ, ws.PTmp)
        ws.oldP .-= vP1
        if norm(ws.oldP) < length(va)*eps()
            steady = true
        end
    end
    return steady
end

function full_update!(va1, va, vatt, vd, vcholF, vK, vP1, vP, vPtt, vT, vv, vZP, d, steady, ws)
    # K = iF*Z*P
    get_K!(vK, vZP, vcholF)
    # att = a + K'*v
    get_updated_a!(vatt, va, vK, vv)
    if !steady
        # Ptt = P - K'*Z*P
        get_updated_Ptt!(vPtt, vP, vK, vZP)
    end
    steady = simple_update!(va1, vatt, vd, vP1, vPtt, vT, steady, ws)
end

function kalman_filter!(Y::AbstractArray{X},
                        c::AbstractArray{U},
                        Z::AbstractArray{W},
                        H::AbstractArray{U},
                        d::AbstractArray{U},
                        T::AbstractArray{U},
                        R::AbstractArray{U},
                        Q::AbstractArray{U},
                        a::AbstractArray{U},
                        att::AbstractArray{U},
                        P::AbstractArray{U},
                        Ptt::AbstractArray{U},
                        start::V,
                        last::V,
                        presample::V,
                        ws::KalmanWs,
                        data_pattern::Vector{Vector{V}}) where {U <: Real, W <: Real, V <: Integer, X <: Union{Real, Missing}}
    changeC = ndims(c) > 1
    changeH = ndims(H) > 2
    changeD = ndims(d) > 1
    changeT = ndims(T) > 2
    changeR = ndims(R) > 2
    changeQ = ndims(Q) > 2
    changeA = ndims(a) > 1
    changeP = ndims(P) > 2
    changeK = ndims(ws.K) > 2
    changeF = ndims(ws.F) > 2
    changeiFv = ndims(ws.iFv) > 1
    ny = size(Y, 1)
    nobs = last - start + 1
    ns = size(T,1)
    # QQ = R*Q*R'
    vR = view(R, :, :, 1)
    vQ = view(Q, :, :, 1)
    get_QQ!(ws.QQ, vR, vQ, ws.RQ)
    l2pi = log(2*pi)
    t = start
    steady = false
    vP = view(P, :, :, 1)
    copy!(ws.oldP, vP)
    cholHset = false
    while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, :, t) : view(c, :)
        #        ws.csmall .= view(vc, pattern)
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, Z, pattern, ndata, ny, t)
        vH = changeH ? view(H, :, :, t) : view(H, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
        va = changeA ? view(a, :, t) : view(a, :)
        vatt = changeA ? view(att, :, t) : view(att, :)
        va1 = changeA ? view(a, :, t + 1) : view(a, :)
        vd = changeD ? view(d, :, t) : view(d, :)
        vP = changeP ? view(P, :, :, t) : view(P, :, :)
        vPtt = changeP ? view(Ptt, :, :, t) : view(Ptt, :, :)
        vP1 = changeP ? view(P, :, :, t + 1) : view(P, :, :)
        vK = changeK ? view(ws.K, 1:ndata, :, t) : view(ws.K, 1:ndata, :)
        if changeR || changeQ
            get_QQ!(ws.QQ, vR, vQ, ws.RQ)
        end
        viFv = changeiFv ? view(ws.iFv, 1:ndata, t) : view(ws.iFv, 1:ndata)
            
        vv = view(ws.v, 1:ndata, t)
        vF = changeF ? view(ws.F, 1:ndata, 1:ndata, t) : view(ws.F, 1:ndata, 1:ndata)
        vvH = view(vH, pattern, pattern)
        vZP = view(ws.ZP, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata, t)
        vcholH = view(ws.cholH, 1:ndata, 1:ndata)
    
        if ndata == 0
            # no observation this period
            if t <= last
                changeA && copy!(vatt, va)
                changeP && copy!(vPtt, vP)
                steady = simple_update!(va1, va, vd, vP1, vP, vT, steady, ws)
            end
            t = t + 1
            continue
        end
        # some observations this period
        # v  = Y[:,t] - c - Z*a
        get_v!(vv, Y, vc, vZsmall, va, t, pattern)
        # F  = Z*P*Z' + H
        get_F!(vF, vZP, vZsmall, vP, vvH)
        info = get_cholF!(vcholF, vF)
        if t <= start + 6
            @show t, det(vF), info
        end
        if info != 0
            # F is near singular
            if !cholHset
                get_cholF!(vcholH, vvH)
                cholHset = true
            end
            ws.lik[t] = ndata*l2pi + univariate_step!(vatt, va1, vPtt, vP1, Y, t, c, ws.Zsmall, vvH, d, T, ws.QQ, va, vP, ws.kalman_tol, ws, pattern)
            t <= start + 5 && @show ws.lik[t] - ndata*l2pi, va1
            t += 1
            continue
        end
        # iFv = inv(F)*v
        get_iFv!(viFv, vcholF, vv)
        ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
        # don't update in last period
        if t < last
            full_update!(va1, va, vatt, vd, vcholF, vK, vP1, vP, vPtt, vT, vv, vZP, d, steady, ws)            
            if t == 5
                @show vatt
                @show va1
                @show vPtt[1,:]
                @show vP1[1, :]
                vattZ = similar(vatt)
                va1Z = similar(va1)
                vPttZ = similar(vPtt)
                vP1Z = similar(vP1)
                univariate_step!(vattZ, va1Z, vPttZ, vP1Z, Y, t, c, ws.Zsmall, vvH, d, T, ws.QQ, va, vP, ws.kalman_tol, ws, pattern)
                @show vattZ
                @show va1Z
                @show vPttZ[1,:]
                @show vP1Z[1, :]
            end
            if changeP && steady && t > 1
                # steady state: covariances are just copied over 
                copy!(vP1, vP)
                vPtt_1 = view(Ptt, :, : , t-1)
                copy!(vPtt, vPtt_1)
            end
        end
        t += 1
    end
    vlik = view(ws.lik, start + presample:last)
    return -0.5*sum(vlik)
end

struct DiffuseKalmanFilterWs{T, U} <: KalmanWs{T, U}
    csmall::Vector{T}
    Zsmall::Matrix{T}
    iZsmall::Vector{U}
    QQ::Matrix{T}
    RQ::Matrix{T}
    c::Vector{T}
    v::Matrix{T}
    F::Matrix{T}
    iF::Matrix{T}
    iFv::Matrix{T}
    a1::Vector{T}
    cholF::Array{T}
    cholH::Matrix{T}
    ZP::Matrix{T}
    Fstar::Matrix{T}
    ZPstar::Matrix{T}
    Kinf::Matrix{T}
    iFZ::Matrix{T}
    K::Matrix{T}
    PTmp::Matrix{T}
    oldP::Matrix{T}
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
        c = Vector{T}(undef, ny)
        v = Matrix{T}(undef, ny, nobs)
        F = Matrix{T}(undef, ny, ny)
        iF = Matrix{T}(undef, ny,ny )
        iFv = Matrix{T}(undef, ny, nobs)
        a1 = Vector{T}(undef, ns)
        cholF = Array{T}(undef, ny, ny, nobs)
        cholH = Matrix{T}(undef, ny, ny)
        ZP = Matrix{T}(undef, ny, ns)
        Fstar = Matrix{T}(undef, ny, ny)
        ZPstar = Matrix{T}(undef, ny, ns)
        Kinf = Matrix{T}(undef, ny, ns)
        iFZ = Matrix{T}(undef, ny, ns)
        K = Matrix{T}(undef, ny, ns)
        PTmp = Matrix{T}(undef, ns, ns)
        oldP = Matrix{T}(undef, ns, ns)
        uKinf = Vector{T}(undef, ns)
        uKstar = Vector{T}(undef, ns)
        Kinf_Finf = Vector{T}(undef, ns)
        ystar = Vector{T}(undef, ny)
        Zstar = Matrix{T}(undef, ny, ns)
        Hstar = Matrix{T}(undef, ny, ny)
        PZi = Vector{T}(undef, ns)
        lik = zeros(T, nobs)
        kalman_tol = 1e-12
        new(csmall, Zsmall, iZsmall, QQ, RQ, c, v, F, iF, iFv, a1, cholF, cholH, ZP, Fstar,
            ZPstar, Kinf, iFZ, K, PTmp, oldP, uKinf, uKstar, Kinf_Finf, ystar, Zstar, Hstar,
            PZi, lik, kalman_tol)
    end
end

DiffuseKalmanFilterWs(ny, ns, np, nobs) = DiffuseKalmanFilterWs{Float64, Int64}(ny, ns, np, nobs)

function diffuse_kalman_filter_init!(Y::AbstractArray{X},
                                     c::AbstractArray{U},
                                     Z::AbstractArray{W},
                                     H::AbstractArray{U},
                                     d::AbstractArray{U},
                                     T::AbstractArray{U},
                                     R::AbstractArray{U},
                                     Q::AbstractArray{U},
                                     a::AbstractArray{U},
                                     att::AbstractArray{U},
                                     Pinf::AbstractArray{U},
                                     Pinftt::AbstractArray{U},
                                     Pstar::AbstractArray{U},
                                     Pstartt::AbstractArray{U},
                                     start::V,
                                     last::V,
                                     presample::V,
                                     tol::U,
                                     ws::KalmanWs,
                                     data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat,
                                                                             V <: Integer,
                                                                             W <: Real,
                                                                             X <: Union{AbstractFloat, Missing}}
    changeC = ndims(c) > 1
    changeH = ndims(H) > 2
    changeD = ndims(d) > 1
    changeT = ndims(T) > 2
    changeR = ndims(R) > 2
    changeQ = ndims(Q) > 2
    changeA = ndims(a) > 1
    changePinf = ndims(Pinf) > 2
    changePstar = ndims(Pstar) > 2
    changeFinf = ndims(ws.F) > 2
    changeFstar = ndims(ws.Fstar) > 2
    
    ny = size(Y, 1)
    t = start
    vR = view(R, :, :, 1)
    vQ = view(Q, :, :, 1)
    get_QQ!(ws.QQ, vR, vQ, ws.RQ)
    l2pi = log(2*pi)
    diffuse_kalman_tol = 1e-8
    kalman_tol = 1e-8
    cholHset = false
    while t <= last
        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, :, t) : view(c, :)
        #        ws.csmall .= view(vc, pattern)
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, Z, pattern, ndata, ny, t)
        vH = changeH ? view(H, :, :, t) : view(H, :, :)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
        if changeR || changeQ
            get_QQ!(ws.QQ, vR, vQ, ws.RQ)
        end
        va = changeA ? view(a, :, t) : view(a, :)
        vatt = changeA ? view(att, :, t) : view(att, :)
        va1 = changeA ? view(a, :, t + 1) : view(a, :)
        vd = changeD ? view(d, :, t) : view(d, :)
        vPinf = changePinf ? view(Pinf, :, :, t) : view(Pinf, :, :)
        vPinftt = changePinf ? view(Pinftt, :, :, t) : view(Pinftt, :, :)
        vPinf1 = changePinf ? view(Pinf, :, :, t + 1) : view(Pinf, :, :)
        vPstar = changePstar ? view(Pstar, :, :, t) : view(Pstar, :, :)
        vPstartt = changePstar ? view(Pstartt, :, :, t) : view(Pstartt, :, :)
        vPstar1 = changePstar ? view(Pstar, :, :, t + 1) : view(Pstar, :, :)
        vv = view(ws.v, 1:ndata, t)
        vvH = view(vH, pattern, pattern)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata, t)
        viFv = view(ws.iFv, 1:ndata, t)
        vFinf = changeFinf ? view(ws.F, 1:ndata, 1:ndata, t) : view(ws.F, 1:ndata, 1:ndata)
        vFstar = changeFstar ? view(ws.Fstar, 1:ndata, 1:ndata, t) : view(ws.Fstar, 1:ndata, 1:ndata)
        vZPinf = view(ws.ZP, 1:ndata, :)
        vZPstar = view(ws.ZPstar, 1:ndata, :)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata, t)
        vcholH = changeH ? view(ws.cholH, 1:ndata, 1:ndata, t) : view(ws.cholH, 1:ndata, 1:ndata)
        viFv = view(ws.iFv, 1:ndata, t)
        vKinf = view(ws.Kinf, 1:ndata, :, t)
        vKstar = view(ws.K, 1:ndata, :, t)

        # v  = Y[:,t] - c - Z*a
        get_v!(vv, Y, vc, vZsmall, va, t, pattern)
        # Finf = Z*Pinf*Z'
        get_F!(vFinf, vZPinf, vZsmall, vPinf)
        info = get_cholF!(vcholF, vFinf)
        if info > 0
            if norm(vFinf) < tol
                return t - 1
            else
                vcholF[1] = NaN
                if !cholHset
                    get_cholF!(vcholH, vvH)
                    cholHset = true
                end
                ws.lik[t] += ndata*l2pi + univariate_step(vatt, va1, vPinftt, vPinf1, vPstartt, vPstar1, Y, t, vc, vZsmall, vvH, vd, vT, ws.QQ, va, vPinf, vPstar, diffuse_kalman_tol, kalman_tol, ws, pattern)
            end
        else
            ws.lik[t] = ndata*l2pi + log(det_from_cholesky(vcholF))
            # Kinf   = iFinf*Z*Pinf                                   %define Kinf'=T^{-1}*K_0 with M_{\infty}=Pinf*Z'
            copy!(vKinf, vZPinf)
            LAPACK.potrs!('U', vcholF, vKinf)
            # Fstar  = Z*Pstar*Z' + H;                                        %(5.7) DK(2012)
            get_F!(vFstar, vZPstar, vZsmall, vPstar, vH)
            # Kstar  = iFinf*(Z*Pstar - Fstar*Kinf)                           %(5.12) DK(2012);
            # note that there is a typo in DK (2003) with "+ Kinf" instead of "- Kinf",
            # but it is correct in their appendix
            get_Kstar!(vKstar, vZsmall, vPstar, vFstar, vKinf, vcholF)
            # att = a + Kinf*v                                                (5.13) DK(2012)
            get_updated_a!(vatt, va, vKinf, vv)
            # a1 = d + T*att                                                  (5.13) DK(2012) 
            update_a!(va1, vd, vT, vatt)
            # Pinf_tt = Pinf - Kinf'*Z*Pinf                                    %(5.14) DK(2012)
            get_updated_Ptt!(vPinftt, vPinf, vKinf, vZPinf)
            # Pinf = T*Ptt*T'
            update_P!(vPinf1, vT, vPinftt, ws.PTmp)
            # Pstartt = Pstar-Pstar*Z'*Kinf-Pinf*Z'*Kstar                           %(5.14) DK(2012)
            get_updated_Pstartt!(vPstartt, vPstar, vZPstar, vKinf, vZPinf,
                                 vKstar, vPinftt, ws.PTmp)
            # Pstar = T*Pstartt*T' + QQ
            update_P!(vPstar1, vT, vPstartt, ws.QQ, ws.PTmp)
            if norm(vPinf1) < tol
                return t
            end
        end
        t += 1
    end
    return t
end

function diffuse_kalman_filter!(Y::AbstractArray{X},
                                c::AbstractArray{U},
                                Z::AbstractArray{W},
                                H::AbstractArray{U},
                                d::AbstractArray{U},
                                T::AbstractArray{U},
                                R::AbstractArray{U},
                                Q::AbstractArray{U},
                                a::AbstractArray{U},
                                att::AbstractArray{U},
                                Pinf::AbstractArray{U},
                                Pinftt::AbstractArray{U},
                                Pstar::AbstractArray{U},
                                Pstartt::AbstractArray{U},
                                start::V,
                                last::V,
                                presample::V,
                                tol::U,
                                ws::KalmanWs,
                                data_pattern::Vector{Vector{V}}) where {U <: AbstractFloat,
                                                                        V <: Integer,
                                                                        W <: Real,
                                                                        X <: Union{AbstractFloat, Missing}}
    ny = size(Y,1)
    nobs = last - start + 1
    get_QQ!(ws.QQ, R, Q, ws.RQ)
    t = diffuse_kalman_filter_init!(Y, c, Z, H, d, T, R, Q, a, att, Pinf, Pinftt, Pstar, Pstartt, start, last, presample, tol, ws, data_pattern)
    kalman_filter!(Y, c, Z, H, d, T, R, Q, a, att, Pstar, Pstartt, t + 1, last, presample, ws, data_pattern)
    vlik = view(ws.lik, start + presample:last)
    return -0.5*sum(vlik)
end

function diffuse_kalman_filter!(Y::AbstractArray{X},
                                c::AbstractArray{U},
                                Z::AbstractArray{W},
                                H::AbstractArray{U},
                                d::AbstractArray{U},
                                T::AbstractArray{U},
                                R::AbstractArray{U},
                                Q::AbstractArray{U},
                                a::AbstractArray{U},
                                att::AbstractArray{U},
                                Pinf::AbstractArray{U},
                                Pinftt::AbstractArray{U},
                                Pstar::AbstractArray{U},
                                Pstartt::AbstractArray{U},
                                start::V,
                                last::V,
                                presample::V,
                                tol::U,
                                ws::KalmanWs) where {U <: AbstractFloat,
                                                     V <: Integer,
                                                     W <: Real,
                                                     X <: Union{AbstractFloat, Missing}}

    m, n = size(Y)
    full_data_pattern = [collect(1:m) for i = 1:n]
    lik = diffuse_kalman_filter!(Y, c, Z, H, d, T, R, Q, a, att, Pinf, Pinftt, Pstar, Pstartt, start, last, presample, tol, ws, full_data_pattern)
    return lik
end
