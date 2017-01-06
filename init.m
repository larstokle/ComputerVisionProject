function [T,S] = init(img0,img1,K)
    
    % Dependencies
    addpath('init_dependencies/8point/');
    addpath('init_dependencies/triangulation/');
    addpath('init_dependencies/essential_matrix/');
    addpath('init_dependencies/cam_projection/');
    addpath('init_dependencies/harris/');
    addpath('init_dependencies/bundle_adjustment/');
    
    debug = false;
    bundle_adjust = false;
    reprojection_error_tolerance = 10; %px
    corner_distance_limit = 120; %px - shoudl be tuned for each set of start frames
        
    %% Extract keypoints and correspondences using harris detector from ex3
    % Harris parameters
    harris_patch_size = 9;
    harris_kappa = 0.08;
    num_keypoints = 2000;
    nonmaximum_supression_radius = 8;
    descriptor_radius = 9;
    match_lambda = 7;
    
    harrisScore0 = harris(img0, harris_patch_size, harris_kappa);
    harrisScore1 = harris(img1, harris_patch_size, harris_kappa);

    corners0 = selectKeypoints(harrisScore0, num_keypoints, nonmaximum_supression_radius);
    corners1 = selectKeypoints(harrisScore1, num_keypoints, nonmaximum_supression_radius);

    descriptors0 = describeKeypoints(img0, corners0, descriptor_radius);
    descriptors1 = describeKeypoints(img1, corners1, descriptor_radius);
    
    % (row,col) -> (x,y)
    corners0 = flipud(corners0);
    corners1 = flipud(corners1);
            
    % Argument order: query, db, match lambda
    matches = matchDescriptors(descriptors1, descriptors0, match_lambda);

    [~, ind1, ind0] = find(matches);
    [~, ind1non, ind0non] = find(matches == 0);

    potential_keypoints = corners1(:,ind1non);
    potential_descriptors = descriptors1(:,ind1non);
    
    p0 = corners0(:,ind0);
    p1 = corners1(:,ind1);
    
    descriptors1 = descriptors1(:,ind1);
    
    if debug
        disp(['Num matches before distance filtering:' num2str(length(p0))])
    end
    
    % Filter matches with to large distance
    acceptedMatchIdx = find(sum((p0 - p1).^2,1) < corner_distance_limit^2);
    p0 = p0(:,acceptedMatchIdx);
    p1 = p1(:,acceptedMatchIdx);
    descriptors1 = descriptors1(:,acceptedMatchIdx);
    
    p0 = [p0 ; ones(1,length(p0))];
    p1 = [p1 ; ones(1,length(p1))];
    
    inlier_mask = (ones(1,length(p0))>0);
    
    disp(['Num matches after distance filtering:' num2str(length(p0))])
       
    %% Debug plot start
    if debug
        figure(3);clf;

        subplot(2,1,1);
        imshow(img1); hold on;
        plotMatches(matches, flipud(corners1), flipud(corners0));    
        title('All matches');
        hold off;

        subplot(2,1,2);
        imshow(img1); hold on;
        plotMatchVectors(p0,p1);
        title('Matches after distance filtering');
        hold off;
        pause(2);
    end
    %% Debug plot end
    
    iteration = 1;    
    has_error = true;

    while has_error
        p0 = p0(:,inlier_mask);
        p1 = p1(:,inlier_mask);
        descriptors1 = descriptors1(:,inlier_mask);
        
        %% Estimate the essential matrix E using the 8-point algorithm with RANSAC

        disp('Estimating Essential matrix with RANSAC..')
        
        [E,inlier_mask] = estimateEssentialMatrix_RANSAC(p0, p1, K, K,[]);

        disp('Essential matrix estimate finished')
        disp(['RANSAC result -- Iteration: ' num2str(iteration) ' Num points: ' num2str(size(p0,2)) ' Num inliers:' num2str(nnz(inlier_mask)) ]);

        %% Extract the relative camera positions (R,T) from the essential matrix
        p0 = p0(:,inlier_mask);
        p1 = p1(:,inlier_mask);
        descriptors1 = descriptors1(:,inlier_mask);
        
        %% Debug plot start
        if debug
            figure(3);clf;

            subplot(2,1,1);
            imshow(img1); hold on;
            plotMatches(matches, flipud(corners1), flipud(corners0));    
            title('All matches');
            hold off;

            subplot(2,1,2);
            imshow(img1); hold on;
            plotMatchVectors(p0,p1);
            title(['Inlier matches after ransac iteration ' num2str(iteration)]);
            hold off;
            pause(2);
        end
        %% Debug plot end        

        % Obtain extrinsic parameters (R,t) from E
        [Rots,u3] = decomposeEssentialMatrix(E);

        % Disambiguate among the four possible configurations
        [R_C1_W,T_C1_W] = disambiguateRelativePose(Rots,u3,p0,p1,K,K);

        %% Triangulate a point cloud using the transformation (R,T)

        % Projection matrices
        M0 = K * eye(3,4);
        M1 = K * [R_C1_W, T_C1_W];

        P = linearTriangulation(p0,p1,M0,M1);

        %% Improve estimate by removing reprojection outliers
        err1 = calculateReprojectionError(P,p0,M0);
        err2 = calculateReprojectionError(P,p1,M1);
        e = err1 + err2;
        inlier_mask = (e < reprojection_error_tolerance);
        positive_z_mask = (P(3,:) > 0);
        inlier_mask = inlier_mask & positive_z_mask;
        
        if nnz(inlier_mask) == numel(inlier_mask)
            has_error = false;
        end
        
        if nnz(inlier_mask) < 8
            disp('WARNING: Number of inliers below 8 in init. Aborting.');
            return
        end
        
        iteration = iteration + 1;
    end
    
    % TODO: Might be redundant - double check loop
    p0 = p0(:,inlier_mask);
    p1 = p1(:,inlier_mask);
    descriptors1 = descriptors1(:,inlier_mask);

    disp(max(sum((p0 - p1).^2,1)));
    
    P = linearTriangulation(p0,p1,M0,M1);
        
    % Debug prior to BA
    if debug
        idx = P(3,:) >= 0;
        plotReprojection(P(:,idx),M0,p0(:,idx),K,img0);
        title('Reprojection frame 0 before BA')
        plotReprojection(P(:,idx),M1,p1(:,idx),K,img1);
        title('Reprojection frame 1 before BA');
        
        % Check reprojection error
        err1 = calculateReprojectionError(P,p0,M0);
        err2 = calculateReprojectionError(P,p1,M1);
        e = err1.^2 + err2.^2;
        
        disp(['Reprojection error before BA: ' num2str(sum(e))])
        
        pause(0.01);
    end
    
    H1 = [R_C1_W , T_C1_W ; 0 0 0 1]^-1; % From C1 to world, i.e. inverse of the 8-p-result
    
    %% Refine estimate using reprojection error
    if bundle_adjust
       H0 = eye(4)^-1;
        
       [hidden_state, observations] = buildBAvectors(P,H0,H1,p0,p1);
       
       disp('Starting bundle adjustment..')
       hidden_state = runBA(hidden_state, observations, K);
       disp('Bundle adjustment finished')
       
       [P,H0,H1] = unwrapHiddenState(hidden_state);
       
       % Invert st. transformations are World to Ci
       H0 = H0^-1;
       H1 = H1^-1;
              
       % Make world frame equal to frame 0
       P = H0*P; %Now        
       H1 = H1*H0^-1; % This is World to C1 now
       
       if debug
           R1 = H1(1:3,1:3);
           t1 = H1(1:3,4);
           M0 = K * eye(3,4);
           M1 = K * [R1,t1];
           idx = P(3,:) >= 0;
           plotReprojection(P(:,idx),M0,p0(:,idx),K,img0);
           title('Reprojection frame 0 after BA')
           plotReprojection(P(:,idx),M1,p1(:,idx),K,img1);
           title('Reprojection frame 1 after BA')           

           % Check reprojection error
           err1 = calculateReprojectionError(P,p0,M0);
           err2 = calculateReprojectionError(P,p1,M1);
           e = err1.^2 + err2.^2;

           disp(['Reprojection error after BA: ' num2str(sum(e))])           
       end
       
       H1 = H1^-1; % Convert from World->C1 into C1->World
       
    end
    
    %% Set return values
    
    % Pose of last frame. Transformation from camera to world coordinates
    T = H1; % Still from C1 to world
    
    assert(size(P(:,1),1)==4,'P should be a 4-vector')
    
    % Initial state S^{i1} with first set of 2D-3D correspondences
    
    potentialBearing = K\[potential_keypoints; ones(1, size(potential_keypoints,2))];
    potentialBearing = T(1:3,1:3)*(potentialBearing./(ones(3,1)*sqrt(sum(potentialBearing.^2,1))));
    
    % Used in tracker
    
    dummyRot = [ T(1:3,1:3), zeros(3,1) ; 0 0 0 1];
    dummyRotTwist = HomogMatrix2twist(dummyRot)/2; % Half the magnitude of rotation
    dummyRot = twist2HomogMatrix(dummyRotTwist);
    dummypose = [dummyRot(1:3,1:3) , T(1:3,4)./2 ; 0 0 0 1]; % Half the translation
    
    state.poses = [dummypose(:) , T(:)];
    state.establishedKeypoints = p1(1:2,:);
    state.establishedDescriptors = descriptors1;
    state.establishedDescriptorsTransform = [zeros(4,size(state.establishedKeypoints,2)); state.establishedKeypoints];
    state.landmarks = P;

    state.potentialKeypoints = potential_keypoints;
    state.potentialDescriptors = potential_descriptors;
    state.potentialBearingFirst = potentialBearing; % TODO: Test this
    state.potentialKeypointsFirst = potential_keypoints;
    state.potentialPoseIndFirst = 2*ones(1,size(potential_keypoints,2));
    
    S = state;
end

function [hidden_state,observations] = buildBAvectors(P,H1,H2,p1,p2) 
    n = 2; % num frames
    m = length(P); % num landmarks
    k = length(p1); % num observed landmarks, k1=k2 here
    
    assert(m == k);
    
    % De-homogenize
    p1 = p1(1:2,:);
    p2 = p2(1:2,:);
    
    % (x,y) -> (row,col)
    p1 = flipud(p1);
    p2 = flipud(p2);
    
    % De-homogenize
    P = P(1:3,:);
    
    tau1 = HomogMatrix2twist(H1);
    tau2 = HomogMatrix2twist(H2);
    
    hidden_state = [tau1 ; tau2 ; P(:)];
    O1 = [k ; p1(:) ; (1:k)'];
    O2 = [k ; p2(:) ; (1:k)'];
    observations = [n ; m ; O1 ; O2];
end

function [P,H1,H2] = unwrapHiddenState(hidden_state)
    tau1 = hidden_state(1:6);
    H1 = twist2HomogMatrix(tau1);
    
    tau2 = hidden_state(7:12);
    H2 = twist2HomogMatrix(tau2);    
    
    P = hidden_state(13:end);
    num_P = numel(P)/3;
    P = reshape(P,3,num_P);
    
    % Re-homogenize
    P = [P ; ones(1,num_P)];
end

function [err, inlier_mask] = reprojectionErrorCostFn(F,p1,p2,K)
    
    reprojection_error_tolerance = 4;

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
    inlier_mask = (err < reprojection_error_tolerance);    
end

function err = reprojectionErrorSquared(P,p,M)
    p_proj = M*P; % Project 3D points into image plane using projection matrix M
    p_proj = p_proj./p_proj(3,:); % Dehomogenize
    err = sum((p_proj - p).^2,1); % 
end

function plotReprojection(P,M,p,K,img)

    p_proj = projectPoints(K\M*P,K);
    [p_err, p_err_dir] = calculateReprojectionError(P,p,M);
    
    figure();
    imshow(img); hold on;
    plot(p_proj(1,:), p_proj(2,:), 'ys');
    plot(p(1,:), p(2,:), 'rx');
    quiver(p(1,:), p(2,:),p_err.*p_err_dir(1,:),p_err.*p_err_dir(2,:),0,'-g');
    hold off;
end
