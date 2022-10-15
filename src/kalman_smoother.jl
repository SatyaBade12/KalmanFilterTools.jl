struct KalmanSmootherWs{T, U} <: KalmanWs{T, U}
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
    r_1::Vector{T}
    at_t::Matrix{T}
    K::Array{T}
    KDK::Array{T}
    L::Matrix{T}
    L1::Matrix{T}
    N::Matrix{T}
    N_1::Matrix{T}
    ZP::Matrix{T}
    Kv::Matrix{T}
    iFZ::Matrix{T}
    PTmp::Matrix{T}
    oldP::Matrix{T}
    lik::Vector{T}
    KT::Matrix{T}
    D::Matrix{T}
    ystar::Vector{Union{AbstractFloat, Missing}}
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
    
    function KalmanSmootherWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
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
        iFv = Array{T}(undef, ny, nobs)
        r = zeros(T, ns)
        r_1 = zeros(T, ns)
        at_t = zeros(T, ns, nobs)
        K = Array{T}(undef, ny, ns, nobs)
        KDK = Array{T}(undef, ns, ny, nobs)
        L = Matrix{T}(undef, ns, ns)
        L1 = Matrix{T}(undef, ns, ns)
        N = zeros(T, ns, ns)
        N_1 = zeros(T, ns, ns)
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
            iFv, a1, r, r_1, at_t, K, KDK, L, L1, N, N_1, ZP, Kv,
            iFZ, PTmp, oldP, lik, KT, D, ystar, Zstar, Hstar, PZi,
            tmp_np, tmp_ns, tmp_ny, tmp_ns_np, tmp_ny_ny, tmp_ny_ns,
            kalman_tol)
    end
end

KalmanSmootherWs(ny, ns, np, nobs) = KalmanSmootherWs{Float64, Int64}(ny, ns, np, nobs)

function kalman_smoother!(Y::AbstractArray{V},
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
                          alphah::AbstractArray{U},
                          epsilonh::AbstractArray{U},
                          etah::AbstractArray{U},
                          Valpha::AbstractArray{U},
                          Vepsilon::AbstractArray{U},
                          Veta::AbstractArray{U},
                          start::X,
                          last::X,
                          presample::X,
                          ws::KalmanWs,
                          data_pattern::Vector{Vector{X}}) where {U <: AbstractFloat, V <: Union{AbstractFloat, Missing}, W <: Real, X <: Integer}

    changeC = ndims(c) > 1
    changeH = ndims(H) > 2
    changeD = ndims(d) > 1
    changeT = ndims(T) > 2
    changeR = ndims(R) > 2
    changeQ = ndims(Q) > 2
    changeA = ndims(a) > 1
    changeAtt = ndims(att) > 1
    changeP = ndims(P) > 2
    changePtt = ndims(Ptt) > 2

    kalman_filter!(Y,c, Z, H, d, T, R, Q, a, att,
                   P, Ptt, start, last, presample, ws,
                   data_pattern)
    fill!(ws.r_1,0.0)
    fill!(ws.N_1,0.0)

    ny = size(Y, 1)
    for t = last: -1: start
        #inputs
        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, :, t) : view(c, :)
        #        ws.csmall .= view(vc, pattern)
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, Z, pattern, ndata, ny, t)
        vH = changeH ? view(H, pattern, pattern, t) : view(H, pattern, pattern)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        va = changeA ? view(a, :, t) : view(a, :)
        vatt = changeAtt ? view(att, :, t) : view(att, :)
        vP = changeP ? view(P, :, :, t) : view(P, :, :)
        vPtt = changePtt ? view(Ptt, :, :, t) : view(Ptt, :, :)
        vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vK = view(ws.K, 1:ndata, :, t)
        vKDK = view(ws.KDK, :, 1:ndata, 1) # amounts to K_t (4.22): here KDK = T*K'
        vcholF = view(ws.cholF, 1:ndata, 1:ndata, t)
        viF = view(ws.iF, 1:ndata, 1:ndata, t)
        viFv = view(ws.iFv, 1:ndata, t)
        viFZ = view(ws.iFZ, 1:ndata, :)
        mul!(vKDK, vT, transpose(vK))

        # L_t = T - KDK_t*Z (DK 4.29)
        get_L!(ws.L, vT, vKDK, vZsmall)
        # r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t (DK 4.44)
        update_r!(ws.r, vZsmall, viFv, ws.L, ws.r_1)
        if (length(alphah) > 0 ||
            length(epsilonh) > 0 ||
            length(etah) > 0)
            # N_{t-1} = Z_t'iF_t*Z_t + L_t'N_t*L_t (DK 4.44)
            get_iFZ!(viFZ, vcholF, vZsmall)
            update_N!(ws.N, vZsmall, viFZ, ws.L, ws.N_1, ws.PTmp)
        end
        if length(epsilonh) > 0
            vepsilonh = view(epsilonh, pattern, t)
            # epsilon_t = H*(iF_t*v_t - KDK_t'*r_t) (DK 4.69)
            vtmp1 = view(ws.tmp_ny, 1:ndata)
            get_epsilonh!(vepsilonh, vH, viFv, vKDK, ws.r_1, vtmp1, ws.tmp_ns)
            if length(Vepsilon) > 0
                vVepsilon = view(Vepsilon, pattern, pattern, t)
                get_iF!(viF, vcholF)
                # D_t = inv(F_t) + KDK_t'*N_t*KDK_t (DK 4.69)
                vD = view(ws.D, 1:ndata, 1:ndata)
                vtmp1 = view(ws.tmp_ny_ns, 1:ndata, :)
                get_D!(vD, viF, vKDK,  ws.N_1, vtmp1)
                # Vepsilon_t = H - H*D_t*H (DK 4.69)
                vtmp1 = view(ws.tmp_ny_ny, 1:ndata, 1:ndata)
                get_Vepsilon!(vVepsilon, vH, vD, vtmp1)
            end
        end
        if length(etah) > 0
            vetah = view(etah, :, t)
            # eta_t = Q*R'*r_t (DK 4.69)
            get_etah!(vetah, vQ, vR, ws.r_1, ws.tmp_np)
            if length(Veta) > 0
                vVeta = view(Veta, :, :, t)
                # Veta_t = Q - Q*R'*N_t*R*Q (DK 4.69)
                get_Veta!(vVeta, vQ, vR, ws.N_1, ws.RQ, ws.tmp_ns_np)
            end
        end

        if length(alphah) > 0
            valphah = view(alphah, :, t)
            # alphah_t = a_t + P_t*r_{t-1} (DK 4.44)
            get_alphah!(valphah, va, vP, ws.r)
        end

        if length(Valpha) > 0
            vValpha = view(Valpha, :, :, t)
            if t==last
                copy!(vValpha,vP)
            else
                # Valpha_t = P_t - P_t*N_{t-1}*P_t (DK 4.44)
                get_Valpha!(vValpha, vP, ws.N_1, ws.PTmp)
            end
        end

        copy!(ws.r_1, ws.r)
        copy!(ws.N_1, ws.N)
    end
end

struct DiffuseKalmanSmootherWs{T, U} <: KalmanWs{T, U}
    csmall::Vector{T}
    Zsmall::Matrix{T}
    # necessary for Z selecting vector with missing variables
    iZsmall::Vector{U}
    RQ::Matrix{T}
    QQ::Matrix{T}
    v::Matrix{T}
    F::Array{T}
    cholF::Array{T}
    Fstar::Array{T}
    cholH::Matrix{T}
    iF::Array{T}
    iFv::Array{T}
    a1::Vector{T}
    r::Vector{T}
    r_1::Vector{T}
    r1::Vector{T}
    r1_1::Vector{T}
    at_t::Matrix{T}
    K::Array{T}
    Kinf::Array{T}
    KDK::Array{T}
    KDKinf::Array{T}
    L::Matrix{T}
    L1::Matrix{T}
    N::Matrix{T}
    N_1::Matrix{T}
    N1::Matrix{T}
    N1_1::Matrix{T}
    N2::Matrix{T}
    N2_1::Matrix{T}
    ZP::Matrix{T}
    ZPstar::Matrix{T}
    Kv::Matrix{T}
    iFZ::Matrix{T}
    PTmp::Matrix{T}
    oldP::Matrix{T}
    lik::Vector{T}
    KT::Matrix{T}
    D::Matrix{T}
    uKinf::Vector{T}
    uKstar::Vector{T}
    ystar::Vector{Union{AbstractFloat, Missing}}
    Kinf_Finf::Vector{T}
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
    
    function DiffuseKalmanSmootherWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        csmall = Vector{T}(undef, ny)
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        RQ = Matrix{T}(undef, ns, np)
        QQ = Matrix{T}(undef, ns, ns)
        F = Array{T}(undef, ny, ny, nobs)
        cholF = Array{T}(undef, ny, ny, nobs)
        Fstar = Array{T}(undef, ny, ny, nobs)
        cholH = Matrix{T}(undef, ny, ny)
        iF = Array{T}(undef, ny, ny, nobs)
        a1 = Vector{T}(undef, ns)
        v = Matrix{T}(undef, ny, nobs)
        iFv = Array{T}(undef, ny, nobs)
        r = zeros(T, ns)
        r_1 = zeros(T, ns)
        r1 = zeros(T, ns)
        r1_1 = zeros(T, ns)
        at_t = zeros(T, ns, nobs)
        K = Array{T}(undef, ny, ns, nobs)
        Kinf = Array{T}(undef, ny, ns, nobs)
        KDK = Array{T}(undef, ns, ny)
        KDKinf = Array{T}(undef, ns, ny)
        L = Matrix{T}(undef, ns, ns)
        L1 = Matrix{T}(undef, ns, ns)
        N = zeros(T, ns, ns)
        N_1 = zeros(T, ns, ns)
        N1 = zeros(T, ns, ns)
        N1_1 = zeros(T, ns, ns)
        N2 = zeros(T, ns, ns)
        N2_1 = zeros(T, ns, ns)
        Kv = Matrix{T}(undef, ns, nobs)
        PTmp = Matrix{T}(undef, ns, ns)
        oldP = Matrix{T}(undef, ns, ns)
        ZP = Matrix{T}(undef, ny, ns)
        ZPstar = Matrix{T}(undef, ny, ns)
        iFZ = Matrix{T}(undef, ny, ns)
        lik = Vector{T}(undef, nobs)
        KT = Matrix{T}(undef, ny, ns)
        D = Matrix{T}(undef, ny, ny)
        uKinf = Vector{T}(undef, ns)
        uKstar = Vector{T}(undef, ns)
        ystar = Vector{T}(undef, ny)
        Kinf_Finf = Vector{T}(undef, ns)
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

        new(csmall, Zsmall, iZsmall, RQ, QQ, v, F, cholF, Fstar,
            cholH, iF, iFv, a1, r, r_1, r1, r1_1, at_t, K, Kinf,
            KDK, KDKinf, L, L1, N, N_1, N1, N1_1, N2, N2_1, ZP,
            ZPstar, Kv, iFZ, PTmp, oldP, lik, KT, D, uKinf, uKstar,
            ystar, Kinf_Finf, Zstar, Hstar, PZi, tmp_np, tmp_ns,
            tmp_ny, tmp_ns_np, tmp_ny_ny, tmp_ny_ns, kalman_tol)
    end
end

DiffuseKalmanSmootherWs(ny, ns, np, nobs) = DiffuseKalmanSmootherWs{Float64, Int64}(ny, ns, np, nobs)

function diffuse_kalman_smoother_coda!(Y::AbstractArray{V},
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
                                       alphah::AbstractArray{U},
                                       epsilonh::AbstractArray{U},
                                       etah::AbstractArray{U},
                                       Valpha::AbstractArray{U},
                                       Vepsilon::AbstractArray{U},
                                       Veta::AbstractArray{U},
                                       start::X,
                                       last::X,
                                       presample::X,
                                       ws::DiffuseKalmanSmootherWs,
                                       data_pattern::Vector{Vector{X}}) where {U <: AbstractFloat, V <: Union{AbstractFloat, Missing}, W <: Real, X <: Integer}

    changeC = ndims(c) > 1
    changeH = ndims(H) > 2
    changeD = ndims(d) > 1
    changeT = ndims(T) > 2
    changeR = ndims(R) > 2
    changeQ = ndims(Q) > 2
    changeA = ndims(a) > 1
    changeAtt = ndims(att) > 1
    changePinf = ndims(Pinf) > 2
    changePinftt = ndims(Pinftt) > 2
    changePstar = ndims(Pstar) > 2
    changePstartt = ndims(Pstartt) > 2

    # change in names to follow DK notation
    r0 = ws.r
    r0_1 = ws.r_1
    r1 = ws.r1
    r1_1 = ws.r1_1
    L0 = ws.L
    L1 = ws.L1
    N0 = ws.N
    N0_1 = ws.N_1
    N1 = ws.N1
    N1_1 = ws.N1_1
    N2 = ws.N2
    N2_1 = ws.N2_1

    fill!(r0_1, 0.0)
    fill!(r1_1, 0.0)

    ny = size(Y, 1)
    for t = last: -1: 1
        #inputs
        pattern = data_pattern[t]
        ndata = length(pattern)
        vc = changeC ? view(c, :, t) : view(c, :)
        #        ws.csmall .= view(vc, pattern)
        vZsmall = get_vZsmall(ws.Zsmall, ws.iZsmall, Z, pattern, ndata, ny, t)
        vH = changeH ? view(H, pattern, pattern, t) : view(H, pattern, pattern)
        vT = changeT ? view(T, :, :, t) : view(T, :, :)
        va = changeA ? view(a, :, t) : view(a, :)
        vatt = changeAtt ? view(att, :, t) : view(att, :)
        vPinf = changePinf ? view(Pinf, :, :, t) : view(Pinf, :, :)
        vPinftt = changePinftt ? view(Pinftt, :, :, t) : view(Pinftt, :, :)
        vPstar = changePinf ? view(Pstar, :, :, t) : view(Pstar, :, :)
        vPstartt = changePstartt ? view(Pstartt, :, :, t) : view(Pstartt, :, :)
        vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vv = view(ws.v, : ,t)
        vK = view(ws.K, 1:ndata, :, t)
        vKinf = view(ws.Kinf, 1:ndata, :, t)
        vKDKinf = view(ws.KDKinf, :, 1:ndata) # amounts to K_t (5.12): here KDKinf = T*Kinf'
        vKDK = view(ws.KDK, :, 1:ndata) # amounts to K_t (5.12): here KDK = T*K'
        viF = view(ws.iF, 1:ndata, 1:ndata, t)
        vcholF = view(ws.cholF, 1:ndata, 1:ndata, t)
        viFv = view(ws.iFv, 1:ndata, t)

        mul!(vKDKinf, vT, transpose(vKinf))
        mul!(vKDK, vT, transpose(vK))
        if isnan(vcholF[1])
            vFinf =view(ws.F, 1:ndata, 1:ndata, t)
            vFstar =view(ws.Fstar, 1:ndata, 1:ndata, t)
            if (length(alphah) > 0 ||
                length(epsilonh) > 0 ||
                length(etah) > 0)
                univariate_diffuse_smoother_step!(vT, vFinf, vFstar,
                                                  vKinf, vK,
                                                  L0, L1, N0, N1, N2,
                                                  r0, r1, vv, vZsmall,
                                                  ws.kalman_tol, ws)
            else
                univariate_diffuse_smoother_step!(vT, vFinf, vFstar,
                                                  vKinf, vK,
                                                  L0, L1, r0, r1, vv,
                                                  vZsmall, ws.kalman_tol,
                                                  ws)
            end
            if length(epsilonh) > 0
                vepsilonh = view(epsilonh, pattern, t)
                # epsilon_t = -H_t*KDKinf*r0_t         (DK p. 135)
                vtmp1 = view(ws.tmp_ny, 1:ndata)
                get_epsilonh!(vepsilonh, vH, vKDKinf, r0_1, vtmp1)
                if length(Vepsilon) > 0
                    vVepsilon = view(Vepsilon, pattern, pattern,t)
                    vTmp = view(ws.tmp_ny_ns, 1:ndata, :) 
                    # D_t = KDKinf_t'*N0_t*KDKinf_t    (DK p. 135)
                    vD = view(ws.D, 1:ndata, 1:ndata)
                    get_D!(vD, vKDK,  N0_1, vTmp)
                    # Vepsilon_t = H - H*D_t*H         (DK p. 135)
                    vTmp = view(ws.tmp_ny_ny, 1:ndata, 1:ndata)
                    get_Vepsilon!(vVepsilon, vH, vD, vTmp)
                end
            end
            if length(etah) > 0
                vetah = view(etah, :, t)
                # eta_t = Q*R'*r0_t                    (DK p. 135)
                get_etah!(vetah, vQ, vR, r0_1, ws.tmp_np)
                if length(Veta) > 0
                    vVeta = view(Veta, :, :, t)
                    # Veta_t = Q - Q*R'*N0_t*R*Q        (DK p. 135)
                    get_Veta!(vVeta, vQ, vR, N0_1, ws.RQ, ws.tmp_ns_np)
                end
            end

            if length(alphah) > 0
                valphah = view(alphah, :, t)
                # alphah_t = a_t + Pstar_t*r0_{t-1} + Pinf_t*r1_{t-1}     (DK 5.24)
                get_alphah!(valphah, va, vPstar, vPinf, r0, r1)
            end

            if length(Valpha) > 0
                vValpha = view(Valpha, :, :, t)
                # Valpha_t = Pstar_t - Pstar_t*N0_{t-1}*Pstar_t
                #            -(Pinf_t*N1_{t-1}*Pstar_t)'
                #            -Pinf_t*N1_{t-1}*Pstar_t
                #            -Pinf_t*N2_{t-1}*Pinf_t                       (DK 5.30)
                get_Valpha!(vValpha, vPstar, vPinf,
                            N0, N1, N2, ws.PTmp)
            end
            mul!(r0_1, transpose(T), r0)
            mul!(r1_1, transpose(T), r1)
            mul!(ws.PTmp, transpose(T), N0)
            mul!(N0_1, ws.PTmp, T)
            mul!(ws.PTmp, transpose(T), N1)
            mul!(N1_1, ws.PTmp, T)
            mul!(ws.PTmp, transpose(T), N2)
            mul!(N2_1, ws.PTmp, T)
        else
            # iFv = Finf \ v
            get_iFv!(viFv, vcholF, vv)
            # L0_t = T - KDKinf*Z (DK 5.12)
            get_L!(L0, vT, vKDKinf, vZsmall)
            # L1_t = - KDK*Z (DK 5.12)
            get_L1!(L1, vKDK, vZsmall)
            # r1_{t-1} = Z_t'*iFinf_t*v_t + L0_t'r1_t + L1_t'*r0_t  (DK 5.21)
            update_r!(r1, vZsmall, viFv, L0, r1_1, L1, r0_1)
            # r0_{t-1} = L0_t'r0_t (DK 5.21)
            mul!(r0, transpose(L0), r0_1)
            if (length(alphah) > 0 ||
                length(epsilonh) > 0 ||
                length(etah) > 0)
                # N0_{t-1} = L0_t'N0_t*L0_t (DK 5.29)
                update_N0!(N0, L0, N0_1, ws.PTmp)
                # F1 = inv(Finf)
                # N1_{t-1} = Z'*F1*Z + L0'*N1_t*L0 + L1'*N0_t*L0
                viFZ = view(ws.iFZ, 1:ndata, :)
                get_iFZ!(viFZ, vcholF, vZsmall)
                update_N1!(N1, vZsmall, viFZ, L0, N1_1, L1, N0_1, ws.PTmp)
                # F2 = -inv(Finf)*Fstar*inv(Finv)
                # N2_{t-1} = Z'*F2*Z + L0'*N2_t*L0 + L0'*N1_t*L1
                #            + L1'*N1_t*L0 + L1'*N0_t*L1
                vTmp = view(ws.tmp_ny_ns, 1:ndata, :) 
                vFstar =view(ws.Fstar, 1:ndata, 1:ndata, t)
                update_N2!(N2, viFZ, vFstar, L0, N2_1, N1_1,
                           L1, N0_1, vTmp, ws.PTmp)
            end

            if length(epsilonh) > 0
                vepsilonh = view(epsilonh, :, t)
                # epsilon_t = -H_t*KDKinf*r0_t         (DK p. 135)
                get_epsilonh!(vepsilonh, vH, vKDKinf, r0_1, ws.tmp_ny)
                if length(Vepsilon) > 0
                    vVepsilon = view(Vepsilon,:,:,t)
                    vTmp = view(ws.tmp_ny_ns, 1:ndata, :) 
                    # D_t = KDKinf_t'*N0_t*KDKinf_t    (DK p. 135)
                    get_D!(ws.D, vKDK,  N0_1, vTmp)
                    # Vepsilon_t = H - H*D_t*H         (DK p. 135)
                    vTmp = view(ws.tmp_ny_ny, 1:ndata, 1:ndata)
                    get_Vepsilon!(vVepsilon, vH, ws.D, ws.tmp_ny_ny)
                end
            end
            if length(etah) > 0
                vetah = view(etah, :, t)
                # eta_t = Q*R'*r0_t                    (DK p. 135)
                get_etah!(vetah, vQ, vR, r0_1, ws.tmp_np)
                if length(Veta) > 0
                    vVeta = view(Veta, :, :, t)
                    # Veta_t = Q - Q*R'*N0_t*R*Q        (DK p. 135)
                    get_Veta!(vVeta, vQ, vR, N0_1, ws.RQ, ws.tmp_ns_np)
                end
            end

            if length(alphah) > 0
                valphah = view(alphah, :, t)
                # alphah_t = a_t + Pstar_t*r0_{t-1} + Pinf_t*r1_{t-1}     (DK 5.24)
                get_alphah!(valphah, va, vPstar, vPinf, r0, r1)
            end

            if length(Valpha) > 0
                vValpha = view(Valpha, :, :, t)
                # Valpha_t = Pstar_t - Pstar_t*N0_{t-1}*Pstar_t
                #            -(Pinf_t*N1_{t-1}*Pstar_t)'
                #            -Pinf_t*N1_{t-1}*Pstar_t
                #            -Pinf_t*N2_{t-1}*Pinf_t                       (DK 5.30)
                get_Valpha!(vValpha, vPstar, vPinf,
                            N0, N1, N2, ws.PTmp)
            end
            copy!(r1_1, r1)
            copy!(r0_1, r0)
            copy!(N0_1, N0)
            copy!(N1_1, N1)
            copy!(N2_1, N2)
        end
    end
end

function diffuse_kalman_smoother!(Y::AbstractArray{X},
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
                                  alphah::AbstractArray{U},
                                  epsilonh::AbstractArray{U},
                                  etah::AbstractArray{U},
                                  Valphah::AbstractArray{U},
                                  Vepsilonh::AbstractArray{U},
                                  Vetah::AbstractArray{U},
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
    t = diffuse_kalman_filter_init!(Y, c, Z, H, d, T, R, Q, a, att,
                                    Pinf, Pinftt, Pstar, Pstartt,
                                    start, last, presample, tol, ws,
                                    data_pattern)
    kalman_smoother!(Y, c, Z, H, d, T, R, Q, a, att, Pstar, Pstartt,
                     alphah, epsilonh, etah, Valphah, Vepsilonh,
                     Vetah, t + 1, last, presample, ws, data_pattern)
    diffuse_kalman_smoother_coda!(Y, c, Z, H, d, T, R, Q, a, att,
                                  Pinf, Pinftt, Pstar, Pstartt,
                                  alphah, epsilonh, etah, Valphah,
                                  Vepsilonh, Vetah, start, t,
                                  presample, ws, data_pattern)
    vlik = view(ws.lik, start + presample:last)
    return -0.5*sum(vlik)
end

function diffuse_kalman_smoother!(Y::AbstractArray{X},
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
                                  alphah::AbstractArray{U},
                                  epsilonh::AbstractArray{U},
                                  etah::AbstractArray{U},
                                  Valphah::AbstractArray{U},
                                  Vepsilonh::AbstractArray{U},
                                  Vetah::AbstractArray{U},
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
    lik = diffuse_kalman_smoother!(Y, c, Z, H, d, T, R, Q, a, att,
                                   Pinf, Pinftt, Pstar, Pstartt,
                                   alphah, epsilonh, etah, Valphah,
                                   Vepsilonh, Vetah, start, last,
                                   presample, tol, ws,
                                   full_data_pattern)
    return lik
end
