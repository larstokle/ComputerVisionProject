function err = reprojectionErrorCostFn(F,p1,p2,K)
    
    %% Get R,t from F

    E = K'*F*K; % Assumes K1=K2=K
    
    % Obtain extrinsic parameters (R,t) from E
    [Rots,u3] = decomposeEssentialMatrix(E);

    % Disambiguate among the four possible configurations
    [R_C2_W,t_C2_W] = disambiguateRelativePose(Rots,u3,p1,p2,K,K);

    %% Triangulate a point cloud using the transformation (R,T)

    % Projection matrices
    M1 = K * eye(3,4);
    M2 = K * [R_C2_W, t_C2_W];

    P = linearTriangulation(p1,p2,M1,M2);    
    
    %% Calculate reprojection error
    err1 = reprojectionErrorSquared(P,p1,M1);
    err2 = reprojectionErrorSquared(P,p2,M2);
    err = sqrt(err1 + err2); 
end

function err = reprojectionErrorSquared(P,p,M)
    p_proj = M*P; % Project 3D points into image plane using projection matrix M
    p_proj = p_proj./p_proj(3,:); % Dehomogenize
    err = sum((p_proj - p).^2,1); % 
end