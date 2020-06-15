using Test
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
    
    function KalmanSmootherWs{T, U}(ny::U, ns::U, np::U, nobs::U) where {T <: AbstractFloat, U <: Integer}
        csmall = Vector{T}(undef, ny)
        Zsmall = Matrix{T}(undef, ny, ns)
        iZsmall = Vector{U}(undef, ny)
        RQ = Matrix{T}(undef, ns, np)
        QQ = Matrix{T}(undef, ns, ns)
        F = Matrix{T}(undef, ny, ny)
        cholF = Matrix{T}(undef, ny, ny)
        cholH = Matrix{T}(undef, ny, ny)
        iF = Array{T}(undef, ny, ny, nobs)
        a1 = Vector{T}(undef, ns)
        v = Matrix{T}(undef, ny, nobs)
        iFv = Array{T}(undef, ny, nobs)
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

KalmanSmootherWs(ny, ns, np, nobs) = KalmanSmootherWs{Float64, Int64}(ny, ns, np, nobs)

function kalman_smoother!(Y::AbstractArray{U},
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
    changeAtt = ndims(att) > 1
    changeP = ndims(P) > 2
    changePtt = ndims(Ptt) > 2
   
    kalman_filter!(Y,c, Z, H, d, T, R, Q, a, att,
                   P, Ptt, start, last, presample, ws,
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
        va = changeA ? view(a, :, t) : view(a, :)
        vatt = changeAtt ? view(att, :, t) : view(att, :)
        vP = changeP ? view(P, :, :, t) : view(P, :, :)
        vPtt = changePtt ? view(Ptt, :, :, t) : view(Ptt, :, :)
        vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
        vR = changeR ? view(R, :, :, t) : view(R, :, :)
        vK = view(ws.K, 1:ndata, :, t)
        vKDK = view(ws.KDK, :, 1:ndata, 1) # amounts to K_t (4.22): here KDK = T*K'
        viF = view(ws.iF, 1:ndata, 1:ndata, t)
        viFv = view(ws.iFv, 1:ndata, t)

        mul!(vKDK, vT, transpose(vK))

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
                get_D!(ws.D, viF, vKDK,  ws.N1, ws.tmp_ny_ns)
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
            # alphah_t = a_t + P_t*r_{t-1} (DK 4.44)
            get_alphah!(valphah, va, vP, ws.r)
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

        copy!(ws.r1,ws.r)
        copy!(ws.N1,ws.N)
    end
end
