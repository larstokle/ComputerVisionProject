function [T, S] = processStereo(img_l,img_r,K, baseline, state)
ax2 = get(2,'CurrentAxes');
ax3 = get(3,'CurrentAxes');
ax5 = get(5,'CurrentAxes');
addpath('continuous_dependencies/all_solns/03_stereo');
assert(sum(size(img_l) == size(img_r)) ~= 0);
debug = true;


harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1000;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
r = descriptor_radius;
match_lambda = 6;
max_corner_dist = 200; 
max_epipolar_dist = 15;

min_disp = 2;
max_max_disp = 60;
patch_size = (2*r+1);

disp_patch_radius = 5;

reprojection_pix_tol = 3;


harrisScore = harris(img_l, harris_patch_size, harris_kappa);
corners = selectKeypoints(harrisScore, num_keypoints, nonmaximum_supression_radius);
cornersInds = sub2ind(size(img_l),corners(1,:),corners(2,:));
[sortedCornersInds,cornersOrderToLinear] = sort(cornersInds);
corners = corners(:,cornersOrderToLinear); % does this ruin the best corners first?

dispMap = getDisparityAtPoints(img_l,img_r, corners, disp_patch_radius, min_disp, max_max_disp);
landmarks = disparityToPointCloud(dispMap, K, baseline, img_l);

corners = corners(:,dispMap(sortedCornersInds) > 0);
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
    
    matches =  matchDescriptorsAckermannConstrained(prev_descriptors, descriptors,...
                                                    prev_corners, corners,...
                                                    match_lambda, K, max_epipolar_dist, max_corner_dist);

%    matches = matchDescriptors(prev_descriptors, descriptors, match_lambda);
    [~, ind_prev, ind_frame] = find(matches);
    tracked_landmarks = prev_landmarks(:,ind_prev);
    
    [H_frame_W, inliers] = ransacLocalization(corners(:,ind_frame), tracked_landmarks, K, reprojection_pix_tol);
    H_W_frame = H_frame_W\eye(4);
    if debug
        cla(ax3);
        imshow(img_l,'Parent',ax3);
        hold(ax3,'on');
        plotMatches(matches,flipud(prev_corners),flipud(corners),ax3);
        scatter(prev_corners(1,ind_prev),prev_corners(2,ind_prev),'y','Parent',ax3)
        scatter(corners(1,ind_frame(inliers)),corners(2,ind_frame(inliers)),'r','Parent',ax3);
        %pause;
    end
    %sort tracked and untracked to keep world frame better
    ind_frame = ind_frame(inliers); 
    ind_prev = ind_prev(inliers);
    tracked_landmarks = tracked_landmarks(:,inliers);
    
    landmarks(:,ind_frame) = [];
    landmarks = eye(3,4)*H_W_frame*[landmarks;ones(1,size(landmarks,2))]; %set them to world frame;
    
    tracked_corners = corners(:,ind_frame);
    tracked_descriptors = descriptors(:,ind_frame);
    
    corners(:,ind_frame) = [];
    descriptors(:,ind_frame) = [];
    
    landmarks = [tracked_landmarks, landmarks];
    corners = [tracked_corners,corners];
    descriptors = [tracked_descriptors,descriptors];
   
else
    H_W_frame = eye(4);
    H_frame_W = eye(4);
end

plotPoseXY(ax2,H_W_frame);
scatter(landmarks(3,:), -landmarks(1,:),'.b', 'Parent', ax2);
axis(ax2,'equal');

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
    plot(ax5,[S.poses(12+3,end-1), S.poses(12+3,end)],-[S.poses(12+2,end-1), S.poses(12+2,end)],'b')
    axis(ax5,'equal');
end
drawnow;
end