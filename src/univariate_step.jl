function univariate_step!(t, Y, Z, H, T, QQ, a, P, kalman_tol, ws)
    ws.vH = changeH ? view(H, :, :, t) : view(H, :, :)
    if isdiag(vH)
        univariate_step_0(y, Z, vH, T, QQ, a, P, kalman_tol, ws)
    else
        copy!(ws.ystar, y)
        transformed_measurement!(ws.ystar, ws.Zstar, ws.Hstar, y, Z, ws.vH, changeH)
        univariate_step_0(ws.ystar, ws.Zstar, ws.Hstar, T, QQ, a, P, kalman_tol, ws)
    end
end

function transformed_measurement!(ystar, Zstar, y, Z, cholH)
    LTcholH = LowerTriangular(cholH)
    copy!(ystar, y)
    ldiv!(LTcholH, ystar)
    copy!(Zstar, Z)
    ldiv!(LTcholH, Zstar)
    detLTcholH = 1
    for i = 1:size(LTcholH,1)
        detLTcholH *= LTcholH[i,i]
    end
    return detLTcholH
end

function univariate_step!(Y, t, Z, H, T, RQR, a, P, kalman_tol, ws)
    ny = size(Y,1)
    detLTcholH = 1.0
    if !isdiag(H)
        detLTcholH = transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
    end
    llik = 0
    for i=1:ny
        Zi = view(ws.Zstar, i, :)
        v = get_v!(ws.ystar, ws.Zstar, a, i)
        F = get_F(Zi, P, H[i,i], ws.PZi)
        if F > kalman_tol
            a .+= (v/F) .* ws.PZi
            # P = P - PZi*PZi'/F 
            ger!(-1.0/F, ws.PZi, ws.PZi, P) 
            llik += log(F) + v*v/F
        end
    end
    mul!(ws.a1, T, a)
    a .= ws.a1
    mul!(ws.PTmp, T, P)
    copy!(P, RQR)
    mul!(P, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*log(detLTcholH)
end

function univariate_step(t, Y, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws)
    llik = 0
    for i=1:size(Y, 1)
        Zi = view(Z, i, :)
        v = get_v!(Y, Z, a, i, t)
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
            llik += log(Fstar) + v*v/Fstar
            a .+= ws.uKstar.*(v/Fstar)
            ger!(-1/Fstar, ws.uKstar, ws.uKstar, Pstar)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
    end
    return llik
end

function univariate_step(t, Y, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws, pattern)
    llik = 0
    for i=1:size(pattern, 1)
        Zi = view(Z, pattern[i], :)
        v = get_v(Y, Z, a, pattern[i], t)
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
            llik += log(Fstar) + v*v/Fstar
            a .+= ws.uKstar.*(v/Fstar)
            ger!(-1/Fstar, ws.uKstar, ws.uKstar, Pstar)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
    end
    return llik
end

