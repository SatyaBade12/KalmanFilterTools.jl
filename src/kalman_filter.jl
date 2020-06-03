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

