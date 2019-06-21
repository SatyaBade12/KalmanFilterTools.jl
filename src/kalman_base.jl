function det_from_cholesky(achol::AbstractMatrix{T}) where T <: AbstractFloat
    x = 1.0
    @inbounds @simd for i = 1:size(achol,1)
        x *= achol[i,i]
    end
    x*x
end

function det_from_cholesky(achol::AbstractMatrix{T}, tol::T) where T <: AbstractFloat
    x = 1.0
    @inbounds @simd for i = 1:size(achol,1)
        a = abs(achol[i,i])
        if a > tol
            x *= achol[i,i]
        end
    end
    x*x
end

# alphah_t = a_t + P_t*r_{t-1}
function get_alphah!(alphah::AbstractVector{T}, a::AbstractVector{T}, P::AbstractArray{T}, r::AbstractVector{T}) where T <: AbstractFloat
    copy!(alphah, a)
    gemm!('N', 'N', 1.0, P, r, 1.0, alphah)
end

function get_cholF!(cholF::AbstractArray{T}, F::AbstractArray{T}) where T <: AbstractFloat
    cholF .= 0.5.*(F .+ transpose(F))
    LAPACK.potrf!('U', cholF)
end

function get_F!(f::AbstractArray{T}, zp::AbstractArray{T}, z::AbstractArray{T}, p::AbstractArray{T}) where T <: AbstractFloat
    mul!(zp, z, p)
    mul!(f, zp, transpose(z))
end

function get_F!(f::AbstractArray{T}, zp::AbstractArray{T}, z::AbstractVector{U}, p::AbstractArray{T}) where {T <: AbstractFloat, U <: Integer}
    zp .= view(p, z, :)
    f .= view(zp, :, z)
end

function get_Fstar!(z::AbstractVector{T}, p::AbstractArray{T}, h::T, ukstar::AbstractVector{T}) where T <: AbstractFloat
    mul!(ukstar, p, z)
    Fstar = BLAS.dot(z, ukstar) + h                 # F_{*,t} in 5.7 in DK (2012), relies on H being diagonal
end

function get_Fstar!(z::U, p::AbstractArray{T}, h::T, ukstar::AbstractVector{T}) where {T <: AbstractFloat, U <: Integer}
    ukstar .= view(p, :, z)
    Fstar = BLAS.dot(z, ukstar) + h                 # F_{*,t} in 5.7 in DK (2012), relies on H being diagonal
end

function get_Finf!(z::AbstractVector{T}, p::AbstractArray{T}, ukinf::AbstractVector{T}) where T <: AbstractFloat
    mul!(ukinf, p, z)
    Finf  = BLAS.dot(z, ukinf)                         # F_{\infty,t} in 5.7 in DK (2012), relies on H being diagonal
end

function get_Finf!(z::U, p::AbstractArray{T}, ukinf::AbstractVector{T}) where {T <: AbstractFloat, U <: Integer}
    ukinf .= view(p, :, z)
    Finf  = BLAS.dot(z, ukinf)                         # F_{\infty,t} in 5.7 in DK (2012), relies on H being diagonal
end
function get_Kstar!(Kstar::AbstractArray{T}, Z::AbstractArray{T}, Pstar::AbstractArray{T}, Fstar::AbstractArray{T}, K::AbstractArray{T}, cholF::AbstractArray{T}) where T <: AbstractFloat
    mul!(Kstar, Z, Pstar)
    gemm!('N', 'N', -1.0, Fstar, K, 1.0, Kstar)
    LAPACK.potrs!('U', cholF, Kstar)
end

function get_Kstar!(Kstar::AbstractArray{T}, z::AbstractVector{U}, Pstar::AbstractArray{T}, Fstar::AbstractArray{T}, K::AbstractArray{T}, cholF::AbstractArray{T}) where {T <: AbstractFloat, U <: Integer}
    Kstar .= view(Pstar, z, :)
    gemm!('N', 'N', -1.0, Fstar, K, 1.0, Kstar)
    LAPACK.potrs!('U', cholF, Kstar)
end

# L = T(I - K'*Z)
function get_L!(L::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, Z::AbstractArray{U}, L1::AbstractArray{U}) where U <: AbstractFloat
    fill!(L1, 0.0)
    for i = 1:size(L1, 1)
        L1[i,i] = 1.0
    end
    gemm!('T', 'N', -1.0, K, Z, 1.0, L1)
    mul!(L, T, L1)
end

# L = T(I - K'*z)
function get_L!(L::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, z::AbstractArray{W}, L1::AbstractArray{U}) where {U <: AbstractFloat, W <: Integer}
    m, n = size(K)
    fill!(L1, 0.0)
    for j = 1:m
        zj = z[j]
        for k=1:n
            L1[k, zj] = -K[j, k]
        end
    end
    for i = 1:n
        L1[i,i] += 1.0
    end
    mul!(L, T, L1)
end

function get_iFv!(iFv::AbstractVector{T}, cholF::AbstractArray{T}, v::AbstractVector{T}) where T <: AbstractFloat
    iFv .= v
    LAPACK.potrs!('U', cholF, iFv)
end

# K = iF*Z*P
function get_K!(K::AbstractArray{T}, ZP::AbstractArray{T}, cholF::AbstractArray{T}) where T <: AbstractFloat
    copy!(K, ZP)
    LAPACK.potrs!('U', cholF, K)
end

function get_QQ!(c::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractMatrix{T}, work::Matrix{T}) where T <: AbstractFloat
    mul!(work, a, b)
    mul!(c, work, transpose(a))
end

function get_prediction_error(Y::AbstractArray{T}, Z::AbstractArray{T}, a::AbstractVector{T}, i::U, t::U) where {T <: AbstractFloat, U <: Integer}
        Zi = view(Z, i, :)
        prediction_error = Y[i, t] - BLAS.dot(Zi, a)          # nu_{t,i} in 6.13 in DK (2012)
end

function get_prediction_error(Y::AbstractArray{T}, z::AbstractVector{U}, a::AbstractVector{T}, i::U, t::U) where {T <: AbstractFloat, U <: Integer}
        prediction_error = Y[i, t] - a[z[i]]                  # nu_{t,i} in 6.13 in DK (2012)
end

# v = y - c - Z*a -- basic
function get_v!(v::AbstractArray{T}, y::AbstractMatrix{T}, c::AbstractArray{T}, z::AbstractArray{T}, a::AbstractArray{T}, iy::U, ny::U) where {T <: AbstractFloat, U <: Integer}
    copyto!(v, 1, y, iy, ny)
    v .-= c
    gemm!('N', 'N', -1.0, z, a, 1.0, v)
end

# v = y - c - a[z] -- Z selection matrix
function get_v!(v::AbstractArray{T}, y::AbstractMatrix{T}, c::AbstractArray{T}, z::AbstractVector{U}, a::AbstractArray{T}, iy::U, ny::U) where {T <: AbstractFloat, U <: Integer}
    copyto!(v, 1, y, iy, ny)
    az = view(a,z)
    v .-= c .+ az
end

# v = y - c - Z*a -- missing observations
function get_v!(v::AbstractArray{T}, y::AbstractMatrix{T}, c::AbstractArray{T}, z::AbstractArray{T}, a::AbstractArray{T}, t::U, pattern::Vector{U}) where {T <: AbstractFloat, U <: Integer}
    v .= view(y, pattern, t) .- view(c, pattern)
    gemm!('N', 'N', -1.0, z, a, 1.0, v)
end

# v = y - c - a[z] -- Z selection matrix and missing variables
function get_v!(v::AbstractArray{T}, y::AbstractMatrix{T}, c::AbstractArray{T}, z::AbstractVector{U}, a::AbstractArray{T}, t::U, pattern::Vector{Int64}) where {T <: AbstractFloat, U <: Integer}
    v .= view(y, pattern, t) .- view(c, pattern) .- view(a, z)
end

# v = y - Z*a -- basic
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{T}, z::AbstractMatrix{T}, a::AbstractVector{T}, iy::U, ny::U) where {T <: AbstractFloat, U <: Integer}
    copyto!(v, 1, y, iy, ny)
    gemv!('N', -1.0, z, a, 1.0, v)
end

# v = y - a[z] -- Z selection matrix
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{T}, z::AbstractVector{U}, a::AbstractVector{T}, iy::U, ny::U) where {T <: AbstractFloat, U <: Integer}
    copyto!(v, 1, y, iy, ny)
    az = view(a,z)
    v .= v .- az
end

# v = y - Z*a -- missing observations
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{T}, z::AbstractMatrix{T}, a::AbstractVector{T}, t::U, pattern::Vector{U}) where {T <: AbstractFloat, U <: Integer}
    v .= view(y, pattern, t)
    gemv!('N', -1.0, z, a, 1.0, v)
end

# v = y - a[z] -- Z selection matrix and missing variables
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{T}, z::AbstractVector{U}, a::AbstractVector{T}, t::U, pattern::Vector{Int64}) where {T <: AbstractFloat, U <: Integer}
    v .= view(y, pattern, t) .- view(a, z)
end

# F = Z*P*Z' + H
function get_F!(f::AbstractArray{T}, zp::AbstractArray{T}, z::AbstractArray{T}, p::AbstractArray{T}, h::AbstractArray{T}) where T <: AbstractFloat
    copy!(f, h)
    mul!(zp, z, p)
    gemm!('N', 'T', 1.0, zp, z, 1.0, f)
end

# F = P(z,z) + H
function get_F!(f::AbstractArray{T}, zp::AbstractArray{T}, z::AbstractVector{I}, p::AbstractArray{T}, h::AbstractArray{T}) where {I <: Integer, T <: AbstractFloat}
    copy!(f, h)
    zp .= view(p, z, :)
    f .+= view(zp, :, z)
end

function get_M!(y::AbstractArray{T}, x::AbstractArray{T}, work::AbstractArray{T}) where T <: AbstractFloat
    copy!(work, x)
    LAPACK.potri!('U', work)
    n = size(x,1)
    # complete lower trianle and change sign of entire matrix
    @inbounds for i = 1:n
        @simd for j = 1:i-1
            y[j, i] = -work[j, i]
        end
        @simd for j = i:n
            y[j, i] = -work[i, j]
        end
    end
end

# V_t = P_t - P_t*N_{t-1}*P_t
function get_V!(V::AbstractArray{T}, P::AbstractArray{T}, N::AbstractArray{T}, Ptmp::AbstractArray{T}) where T <: AbstractFloat
    copy!(V, P)
    mul!(Ptmp, P, N)
    gemm!('N', 'N', -1.0, Ptmp, P, 1.0, V)
end

# a = T(a + K'*v)
function update_a!(a::AbstractVector{U}, K::AbstractMatrix{U}, v::AbstractVector{U}, a1::Vector{U}, T::AbstractMatrix{U}) where U <: AbstractFloat
    copy!(a1, a)
    gemv!('T', 1.0, K, v, 1.0, a1)
    gemv!('N', 1.0, T, a1, 0.0, a)
end

# a = d + T(a + K'*v)
function update_a!(a1::AbstractArray{U}, a::AbstractArray, d::AbstractArray{U}, K::AbstractArray{U}, v::AbstractArray{U}, work::AbstractArray{U}, T::AbstractArray{U}) where U <: AbstractFloat
    copy!(work, a)
    gemm!('T', 'N', 1.0, K, v, 1.0, work)
    gemm!('N', 'N', 1.0, T, work, 0.0, a1)
    a1 .+= d
end

# K = K + Z*W*M*W'
function update_K!(K::AbstractArray{U}, ZWM::AbstractArray{U}, W::AbstractArray{U}) where U <: AbstractFloat
    gemm!('N', 'T', 1.0, ZWM, W, 1.0, K)
end

# M = M + M*W'*Z'iF*Z*W*M
function update_M!(M::AbstractArray{U}, Z::AbstractArray{U}, W::AbstractArray{U}, cholF::AbstractArray{U}, ZW::AbstractArray{U}, ZWM::AbstractArray{U}, iFZWM::AbstractArray{U}) where U <: AbstractFloat
    mul!(ZW, Z, W)
    mul!(ZWM, ZW, M)
    copy!(iFZWM, ZWM)
    LAPACK.potrs!('U', cholF, iFZWM)
    gemm!('T', 'N', 1.0, ZWM, iFZWM, 1.0, M)
end

# M = M + M*W(z,:)'*iF*W(z,:)*M
function update_M!(M::AbstractArray{U}, z::Vector{R}, W::AbstractArray{U}, cholF::AbstractArray{U}, ZW::AbstractArray{U}, ZWM::AbstractArray{U}, iFZWM::AbstractArray{U}) where {U <: AbstractFloat, R <: Real}
    ZW .= view(W, z, :)
    mul!(ZWM, ZW, M)
    copy!(iFZWM, ZWM)
    LAPACK.potrs!('U', cholF, iFZWM)
    gemm!('T', 'N', 1.0, ZWM, iFZWM, 1.0, M)
end

# N_{t-1} = Z_t'iF_t*Z_t + L_t'N_t*L_t
function update_N!(N1::AbstractArray{T}, Z::AbstractArray{T}, iFZ::AbstractArray{T}, L::AbstractArray{T}, N::AbstractArray{T}, Ptmp::AbstractArray{T}) where T <: AbstractFloat
    mul!(N1, transpose(Z), iFZ)
    mul!(Ptmp, transpose(L), N)
    gemm!('N', 'N', 1.0, Ptmp, L, 1.0, N1)
end

function update_N!(N1::AbstractArray{T}, z::AbstractArray{U}, iFZ::AbstractArray{T}, L::AbstractArray{T}, N::AbstractArray{T}, Ptmp::AbstractArray{T}) where {T <: AbstractFloat, U <: Integer}
    fill!(N1, 0.0)
    vN1 = view(N1, z, :)
    vN1 .= iFZ
    mul!(Ptmp, transpose(L), N)
    gemm!('N', 'N', 1.0, Ptmp, L, 1.0, N1)
end

# P = T*(P - K'*Z*P)*T'+ QQ
function update_P!(P::AbstractArray{U}, T::AbstractArray{U}, QQ::AbstractArray{U}, K::AbstractArray{U}, ZP::AbstractArray{U}, Ptmp::AbstractArray{U}) where U <: AbstractFloat
    gemm!('T', 'N', -1.0, K, ZP, 1.0, P) 
    mul!(Ptmp, T, P)
    copy!(P, QQ)
    gemm!('N', 'T', 1.0, Ptmp, T, 1.0, P)
end

# r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t
function update_r!(r1::AbstractVector{T}, Z::AbstractArray{T}, iFv::AbstractVector{T}, L::AbstractArray{T}, r::AbstractVector{T}) where T <: AbstractFloat
    mul!(r1, transpose(Z), iFv)
    gemm!('T', 'N', 1.0, L, r, 1.0, r1)
end

# r_{t-1} = z_t'*iF_t*v_t + L_t'r_t
function update_r!(r1::AbstractVector{T}, z::AbstractVector{U}, iFv::AbstractVector{T}, L::AbstractArray{T}, r::AbstractVector{T}) where {T <: AbstractFloat, U <: Integer}
    fill!(r1, 0.0)
    vr1 = view(r1, z)
    vr1 .= iFv
    gemm!('T', 'N', 1.0, L, r, 1.0, r1)
end

# W = T(W - K'*iF*Z*W)
function update_W!(W::AbstractArray{U}, ZW::AbstractArray{U}, cholF::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, iFZW::AbstractArray{U}, KtiFZW::AbstractArray{U}) where U <: AbstractFloat
    copy!(iFZW, ZW)
    LAPACK.potrs!('U', cholF, iFZW)
    copy!(KtiFZW, W)
    gemm!('T', 'N', -1.0, K, iFZW, 1.0, KtiFZW)
    mul!(W, T, KtiFZW)
end

