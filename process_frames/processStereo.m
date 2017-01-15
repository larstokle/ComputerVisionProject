function [T, S] = processStereo(img_l,img_r,K, baseline, state, plotAx)
%% options 
use_AckermannConstraint = true;
keep_tracked = false;

%% init
mapAx = plotAx.map;
lndmrkAx = plotAx.lndmrk;
heightAx = plotAx.height;

assert(sum(size(img_l) == size(img_r)) ~= 0);
debug = true;

% parameters
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 2000;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
r = descriptor_radius;
match_lambda = 5;
max_corner_dist = 200; 
max_epipolar_dist = 15;

min_disp = 2;
max_max_disp = 60;
patch_size = (2*r+1);

disp_patch_radius = 5;

reprojection_pix_tol = 3;

%% extract keypoints
harrisScore = harris(img_l, harris_patch_size, harris_kappa);
corners = selectKeypoints(harrisScore, num_keypoints, nonmaximum_supression_radius);
cornersInds = sub2ind(size(img_l),corners(1,:),corners(2,:));

%% triangulate
dispMap = getDisparityAtPoints(img_l,img_r, corners, disp_patch_radius, min_disp, max_max_disp);
corners(:,dispMap(cornersInds) == 0 ) = [];
cornersInds = sub2ind(size(img_l),corners(1,:),corners(2,:));
[~, cornersOrderToLinear] = sort(cornersInds);
[~, linearToCornersOrder] = sort(cornersOrderToLinear);
landmarks = disparityToPointCloud(dispMap, K, baseline, img_l);
landmarks = landmarks(:,linearToCornersOrder);

descriptors = describeKeypoints(img_l, corners, descriptor_radius);
corners = flipud(corners);

ok_lndmrks = landmarks(3,:) > 0 & sum(landmarks.^2,1) < 80^2;
landmarks = landmarks(:,ok_lndmrks);
corners = corners(:,ok_lndmrks);
descriptors = descriptors(:,ok_lndmrks);


if  ~isempty(state)
    prev_descriptors = state.descriptors;
    prev_corners = state.keypoints;
    prev_landmarks = state.landmarks;

    %% track keypoints from last frame
    if use_AckermannConstraint
        matches =  matchDescriptorsAckermannConstrained(prev_descriptors, descriptors,...
                                                        prev_corners, corners,...
                                                        match_lambda, K, max_epipolar_dist, max_corner_dist);
    else
        matches = matchDescriptorsLocally(prev_descriptors, descriptors,...
                                          prev_corners, corners,...
                                          match_lambda, max_corner_dist);
    end
    [~, indx_prev, indx_frame] = find(matches);
    tracked_landmarks = prev_landmarks(:,indx_prev);
    
    %% estimate pose
    [H_frame_W, inliers] = ransacLocalization(corners(:,indx_frame), tracked_landmarks, K, reprojection_pix_tol);
    H_W_frame = H_frame_W\eye(4);
    if debug
        cla(lndmrkAx);
        imshow(img_l,'Parent',lndmrkAx);
        hold(lndmrkAx,'on');
        plotMatchVectors(prev_corners(:,indx_prev), corners(:,indx_frame),'r',lndmrkAx);
        plotMatchVectors(prev_corners(:,indx_prev(inliers)), corners(:,indx_frame(inliers)),'g',lndmrkAx);
        title(lndmrkAx,{'Tracked Landmarks';sprintf('#tracked: %i, #inliers: %i, ratio %f',length(indx_frame),sum(inliers),sum(inliers)/length(indx_frame))});
        lh = legend(lndmrkAx,'Outliers','Inliers');
        set(lh,'Location','southoutside','Orientation','horizontal','Box','off');
    end
    
    %% sort tracked and untracked to keep world frame better
    if keep_tracked
        indx_frame = indx_frame(inliers); 
        indx_prev = indx_prev(inliers);
        tracked_landmarks = tracked_landmarks(:,inliers);


        landmarks(:,indx_frame) = [];
        landmarks = eye(3,4)*H_W_frame*[landmarks;ones(1,size(landmarks,2))]; %set them to world frame;

        scatter(landmarks(3,:), -landmarks(1,:),'.c', 'Parent', mapAx);

        tracked_corners = corners(:,indx_frame);
        tracked_descriptors = descriptors(:,indx_frame);

        corners(:,indx_frame) = [];
        descriptors(:,indx_frame) = [];

        landmarks = [tracked_landmarks, landmarks];
        corners = [tracked_corners,corners];
        descriptors = [tracked_descriptors,descriptors];
    else
        landmarks = eye(3,4)*H_W_frame*[landmarks;ones(1,size(landmarks,2))];
        scatter(landmarks(3,indx_frame(inliers)), -landmarks(1,indx_frame(inliers)),'.c', 'Parent', mapAx);
    end
else
    H_W_frame = eye(4);
    H_frame_W = eye(4);
end

plotPoseXY(mapAx,H_W_frame);
axis(mapAx,'equal');

newState.descriptors = descriptors;
newState.keypoints = corners;
newState.landmarks = landmarks;
if ~isempty(state)
    newState.poses = [state.poses,H_W_frame(:)];
else
    newState.poses = H_W_frame(:);
end

T = H_W_frame;

S = newState;

if ~isempty(state)
    plot(heightAx,[S.poses(12+3,end-1), S.poses(12+3,end)],-[S.poses(12+2,end-1), S.poses(12+2,end)],'b')
    axis(heightAx,'equal');
end
drawnow;
end