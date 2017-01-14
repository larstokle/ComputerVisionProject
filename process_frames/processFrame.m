function [pose, state] = processFrame(img, K, H_W0, old_state, plotAx)
%[pose, state] = processFrame(img, K, H_W0, oldState):
%   
%   Input:
%       img:            frame to process
%       K:              camera calibration matrix
%       H_W0:           homogenous transformation to from previous frame to world
%       old_state:      struct
%
%   Output:
%       pose:   pose computed at img relative to world frame
%       state:  updated input struct
% 

%% tracking algorithm options
% Landmarks
use_KLT = false; % tracks landmark descriptors with KLT
use_Ackermann_lndmrk = true; %matches landmarks with ackerman constraint if KLT is not used

% Candidates
use_Ackermann_cndt = true; % matches landmarks with Ackerman constraint
use_epipolarConstr_cndt = false; %matches with epipolar constraint, if use_Ackermann_cndt = false
% otherwise matching with only distance constraint is used

%% Init 
% plotting
mapAx = plotAx.map;
lndmrkAx = plotAx.lndmrk;
cndtAx = plotAx.cndt;

% Tuning parameters frame (new) keypoints
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1500;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;

% Tuning parameters landmarks and pose estimation
KLT_match = 0.001; %fraction of maximum patch distance^2
match_lambda_lndmrk = 7;
max_match_dist_lndmrk = 200;
max_match_projected_dist = 40;
reprojection_pix_tol = 15;
max_epipole_line_dist_lndmrk = 30;

% Tuning parameters candidate keypoints and triangulation
match_lambda_candidates = 7;
max_epipole_line_dist_cndt = 15;
max_match_dist_cndt = 150;
triangulationAngleThresh = 2*pi/180;
triangulationCosThresh = cos(triangulationAngleThresh);
new_landmarks_distance_limit = 60;

% Extract variables from state
poses = old_state.poses;
N_frames = size(poses,2) + 1;

landmarks_w = old_state.landmarks; % w since expressed in world frame

landmark_prev_keypoints = old_state.landmark_keypoints;
landmark_descriptors = old_state.landmark_descriptors;
landmark_transforms = old_state.landmark_transforms;
N_landmarks = size(landmark_prev_keypoints,2);
patch_radius = sqrt(size(landmark_descriptors,1));

candidate_prev_keypoints = old_state.candidate_keypoints;
candidate_descriptors_prev = old_state.candidate_descriptors;
candidate_bearings_first = old_state.candidate_bearings_1;
candidate_keypoints_first = old_state.candidate_keypoints_1;
candidate_pose_idx_first = old_state.candidate_pose_idx_1;
N_candidates = size(candidate_prev_keypoints, 2);

%% find points in this frame to query
harris_score = harris(img, harris_patch_size, harris_kappa);
frame_keypoints = selectKeypoints(harris_score, num_keypoints, nonmaximum_supression_radius);
frame_descriptors = describeKeypoints(img, frame_keypoints, descriptor_radius);
frame_keypoints = flipud(frame_keypoints);

%% find matches in this frame
if use_Ackermann_cndt %using Ackermann or not
    [matches_candidate, theta_frame_hat, varTheta_frame] = matchDescriptorsAckermannConstrained(candidate_descriptors_prev, frame_descriptors, candidate_prev_keypoints, frame_keypoints, match_lambda_candidates, K, max_epipole_line_dist_cndt, max_match_dist_cndt);
    theta_frame_hat = -theta_frame_hat; %maybe not?? -> ackermann estimates theta around upward axis here Y is downward
    fprintf('ACKERMAN: found theta_hat: %f deg\n',theta_frame_hat*180/pi);
else
    matches_candidate = matchDescriptorsLocally(candidate_descriptors_prev, frame_descriptors, candidate_prev_keypoints, frame_keypoints, match_lambda_candidates, max_match_dist_cndt);
end

%% track landmarks points

%estimate new point with same homogenous transform as last transform
H_W_prev = reshape(poses(:,end),4,4);
H_W_prev2 = reshape(poses(:,end-1),4,4);
if size(poses,2) > 2
    H_W_prev3 = reshape(poses(:,end-2),4,4);
else
    H_W_prev3 = (H_W_prev2\H_W_prev)\H_W_prev2; %constant speed first frame =)
end


% Estimate rotation
if ~use_Ackermann_cndt
    R_prev2_prev = H_W_prev2(1:3,1:3)'*H_W_prev(1:3,1:3); %last rotation
    omega_hat_prev2_prev = logm(R_prev2_prev); %last skew
    theta_prev2_prev = norm(matrix2cross(omega_hat_prev2_prev)); %last angle of rotation
    theta_frame_hat = theta_prev2_prev*sign(omega_hat_prev2_prev(2)); 
    fprintf('PREVIOUS FRAME: found theta: %f\n',theta_frame_hat);
end
R_prev_frame_hat = simpleRotY(theta_frame_hat);

% Estimate translation
speed_prev = norm(H_W_prev(1:3,4) - H_W_prev2(1:3,4));
speed_prev2 = norm(H_W_prev2(1:3,4) - H_W_prev3(1:3,4));
accel_prev = speed_prev - speed_prev2;
speed_frame_hat = speed_prev + 0.8*accel_prev; %estimate, not as much accel as last time for stability?
t_prev_frame_hat = simpleRotY(theta_frame_hat/2)*[0; 0; speed_frame_hat];

% Estimate homogenous transform
H_prev_frame_hat = [R_prev_frame_hat, t_prev_frame_hat;
                    0, 0, 0, 1];
H_W_frame_hat = H_W_prev*H_prev_frame_hat;

% Estimate location of landmark keypoints in the (estimated) current camera frame
landmarks_C = H_W_frame_hat\landmarks_w;

if use_KLT
    landmark_transforms(5:6,:) = projectPoints(landmarks_C(1:3,:), K);
else
    landmark_position_estimate = projectPoints(landmarks_C(1:3,:), K);
end

% tracking
if use_KLT    
    % Landmark descriptors are reshaped from vector form to matrix form.
    landmark_descriptors_reshaped = reshape(landmark_descriptors,[patch_radius, patch_radius, N_landmarks]);
    
    % Track landmarks using pyramid KLT
	[transform_KLT, position_KLT, tracked_successfully_KLT] = KLTtracker(img, landmark_transforms, landmark_descriptors_reshaped, KLT_match);
    fprintf('KLT: #landmarks tracked: %i, #landmarks lost: %i\n', sum(tracked_successfully_KLT > 0), sum(tracked_successfully_KLT == 0));
elseif use_Ackermann_lndmrk
    matches_lndmrk = matchDescriptorsAckermannConstrained(landmark_descriptors, frame_descriptors, landmark_prev_keypoints, frame_keypoints, match_lambda_lndmrk,K,max_epipole_line_dist_lndmrk, max_match_dist_lndmrk);
    fprintf('MATCHING: #landmarks tracked: %i #landmarks lost: %i\n', sum(matches_lndmrk > 0), sum(matches_lndmrk == 0));
else
    matches_lndmrk = matchDescriptorsLocally(landmark_descriptors, frame_descriptors, landmark_position_estimate, frame_keypoints, match_lambda_lndmrk, max_match_projected_dist);
    fprintf('MATCHING: #landmarks tracked: %i #landmarks lost: %i\n', sum(matches_lndmrk > 0), sum(matches_lndmrk == 0));
end

%discard nonvalid tracks
if use_KLT
    
    % Tracked landmark position in the current frame
    tracked_landmarks_frame_keypoints = position_KLT(:,tracked_successfully_KLT);
    
    % Reuse old descriptors.
    tracked_landmarks_frame_descriptors = landmark_descriptors(:,tracked_successfully_KLT); 
    
    % Update descriptor transforms
    tracked_landmarks_transforms = transform_KLT(:,tracked_successfully_KLT);
    
    tracked_landmarks = landmarks_w(:,tracked_successfully_KLT);
    
    tracked_landmarks_prev_keypoints = landmark_prev_keypoints(:,tracked_successfully_KLT);
    
    [~, isNew] = setdiff(...
                round(frame_keypoints'/nonmaximum_supression_radius),...
                round(tracked_landmarks_frame_keypoints'/nonmaximum_supression_radius),...
                'rows');
    frame_keypoints = frame_keypoints(:,isNew);
    frame_descriptors = frame_descriptors(:,isNew);
    
else
    [~, establishedInd, current_frame_idx] = find(matches_lndmrk);
    
    tracked_landmarks_frame_keypoints = frame_keypoints(:,current_frame_idx);
    tracked_landmarks_frame_descriptors = frame_descriptors(:,current_frame_idx);
    
    tracked_landmarks_prev_keypoints = landmark_prev_keypoints(:,establishedInd);
    tracked_landmarks = landmarks_w(:,establishedInd);
    
    frame_keypoints(:,current_frame_idx) = []; % = frame_keypoints(:,isNew);
    frame_descriptors(:,current_frame_idx) = []; % = frame_descriptors(:,isNew);
end

% Validation plot landmark keypoints
cla(lndmrkAx);
imshow(img,'Parent',lndmrkAx);
hold(lndmrkAx,'on');

scatter(landmark_prev_keypoints(1,:), landmark_prev_keypoints(2,:), 'yx','Linewidth',2','Parent',lndmrkAx);
plotMatchVectors(tracked_landmarks_prev_keypoints, tracked_landmarks_frame_keypoints, 'r',lndmrkAx);
title('Landmarks tracked from previous frame','Parent',lndmrkAx);

%% Estimate relative pose

[H_CW, inliers] = ransacLocalization(tracked_landmarks_frame_keypoints, tracked_landmarks, K, reprojection_pix_tol);
fprintf('RANSAC: #inliers: %i, #outliers: %i\n',sum(inliers),sum(inliers==0));

tracked_landmarks_frame_keypoints = tracked_landmarks_frame_keypoints(:,inliers);
tracked_landmarks_frame_descriptors = tracked_landmarks_frame_descriptors(:,inliers);

if use_KLT
    tracked_landmarks_transforms = tracked_landmarks_transforms(:,inliers);
end

tracked_landmarks = tracked_landmarks(:,inliers);
tracked_landmarks_prev_keypoints = tracked_landmarks_prev_keypoints(:,inliers);

%validate ransac
plotMatchVectors(tracked_landmarks_prev_keypoints, tracked_landmarks_frame_keypoints, 'g', lndmrkAx);
legend(lndmrkAx,'Landmarks', 'tracked outliers', 'tracked inliers','Location','so','Orientation','horizontal','Box','off');

%% Compute homogenous transforms
H_WC = H_CW\eye(4);     % World to 1
H_prev_C = H_W_prev\H_WC;

%% Track candidate keypoints
% (moved the matching...) should be enough to do it up there! fix removing
% and shit!!
if use_Ackermann_cndt %using Ackermann or not
    [matches_candidate, theta_frame_hat, varTheta_frame] = matchDescriptorsAckermannConstrained(candidate_descriptors_prev, frame_descriptors, candidate_prev_keypoints, frame_keypoints, match_lambda_candidates, K, max_epipole_line_dist_cndt, max_match_dist_cndt);
    %theta_frame_hat = -theta_frame_hat; %ackermann estimates theta around upward axis here Y is downward
elseif use_epipolarConstr_cndt
    matches_candidate = matchDescriptorsEpiPolar(candidate_descriptors_prev, frame_descriptors, candidate_prev_keypoints, frame_keypoints, match_lambda_candidates, H_prev_C, K, max_epipole_line_dist_cndt, max_match_dist_cndt);
else
    matches_candidate = matchDescriptorsLocally(candidate_descriptors_prev, frame_descriptors, candidate_prev_keypoints, frame_keypoints, match_lambda_candidates, max_match_dist_cndt);
end

[~, prev_frame_idx, current_frame_idx] = find(matches_candidate);
tracked_candidate_keypoints = frame_keypoints(:,current_frame_idx);
tracked_candidate_descriptors = frame_descriptors(:,current_frame_idx);
N_candidates_tracked = size(tracked_candidate_keypoints,2);
fprintf('MATCHING: #candidates tracked: %i\n',N_candidates_tracked)

%remove tracked candidate keypoints from set of new candidate keypoints
% New candidate keypoints added at the current frame
frame_keypoints(:,current_frame_idx) = [];
frame_descriptors(:,current_frame_idx) = [];

% Calculate bearing vectors of new candidate keypoints
frame_bearings = calculateBearingVectors(frame_keypoints,H_WC,K);

%validation plot candidates
cla(cndtAx);
imshow(img,'Parent',cndtAx);
hold(cndtAx,'on');
plotMatchVectors(candidate_keypoints_first(:,prev_frame_idx), tracked_candidate_keypoints, 'c', cndtAx)


% Remove nontracked candidates after plotting
candidate_bearings_first = candidate_bearings_first(:,prev_frame_idx);
candidate_keypoints_first = candidate_keypoints_first(:,prev_frame_idx);
candidate_pose_idx_first = candidate_pose_idx_first(:,prev_frame_idx);

%% Triangulate new landmarks

% Calculate bearing vector of candidate keypoint seen from the current
% camera frame
candidate_bearings_curr = calculateBearingVectors(tracked_candidate_keypoints,H_WC,K);

% innerprod = cos(angle)... ==> low innerProd = large angle
inner_product = sum(candidate_bearings_curr.*candidate_bearings_first,1);
can_triangulate = find(inner_product < triangulationCosThresh);

p1_bearings = candidate_bearings_first(:,can_triangulate);
p1_poses_ind = candidate_pose_idx_first(can_triangulate);
p2 = homogenize2D(tracked_candidate_keypoints(:,can_triangulate));

% Triangulate landmarks one by one, since they may have different pose
new_landmarks = zeros(4,size(p2,2));

% Projection matrix for the current frame is fixed
M2 = K*H_CW(1:3,:);

for i= 1:size(p2,2)
    H_iW = reshape(poses(:,p1_poses_ind(i)),4,4)\eye(4);
    M1 = K*H_iW(1:3,:);
    p1 = K*H_iW(1:3,1:3)*p1_bearings(:,i);
    p1 = p1(1:3)/p1(3);
    new_landmarks(:,i) = linearTriangulation(p1, p2(:,i), M1, M2);
end

% Find triangulated points in front of camera and closer than a threshold
new_landmarks_C = H_CW*new_landmarks;

good_new_landmark = find(new_landmarks_C(3,:) > 0 & sqrt(sum(new_landmarks_C(1:3,:).^2,1)) < new_landmarks_distance_limit);

new_landmarks = new_landmarks(:,good_new_landmark);
did_triangulate = can_triangulate(good_new_landmark);

fprintf('TRIANGULATION: #new landmarks: %i, #triangulated: %i, #too close: %i, #too far: %i\n',...
    length(good_new_landmark), length(can_triangulate), sum(new_landmarks_C(3,:) < 0), sum(sqrt(sum(new_landmarks_C(1:3,:).^2,1)) > new_landmarks_distance_limit));

% Update landmark
tracked_landmarks = [tracked_landmarks, new_landmarks];
tracked_landmarks_frame_keypoints = [tracked_landmarks_frame_keypoints, tracked_candidate_keypoints(:,did_triangulate)];
tracked_landmarks_frame_descriptors = [tracked_landmarks_frame_descriptors, tracked_candidate_descriptors(:,did_triangulate)];

% Validation plot new landmarks
%scatter(p2(1,:),p2(2,:),'g','Linewidth',2, 'Parent', cndtAx);
plotMatchVectors(candidate_keypoints_first(:,can_triangulate), tracked_candidate_keypoints(:,can_triangulate), 'r', cndtAx);
plotMatchVectors(candidate_keypoints_first(:,did_triangulate), tracked_candidate_keypoints(:,did_triangulate), 'g', cndtAx);
title({'Tracked candidate keypoints from last frame';sprintf('#tracked = %i, #triangulated = %i, #new landmarks %i',N_candidates_tracked, length(can_triangulate), size(new_landmarks,2))},'Parent',cndtAx);
legend(cndtAx,'tracked','triangulation failed', 'new landmark','Location','so','Orientation','horizontal','Box','off');
scatter(new_landmarks(3,:), -new_landmarks(1,:),'.r', 'Parent', mapAx);
axis(mapAx,'equal');


if use_KLT
    tracked_landmarks_transforms = [tracked_landmarks_transforms,...
        [zeros(4,size(tracked_candidate_keypoints(:,did_triangulate),2));tracked_candidate_keypoints(:,did_triangulate)]];
end
    
not_triangulated = find(inner_product >= triangulationCosThresh);
tracked_candidate_keypoints = tracked_candidate_keypoints(:,not_triangulated);
tracked_candidate_descriptors = tracked_candidate_descriptors(:,not_triangulated);

candidate_bearings_first = candidate_bearings_first(:, not_triangulated);
candidate_keypoints_first = candidate_keypoints_first(:, not_triangulated);
candidate_pose_idx_first = candidate_pose_idx_first(not_triangulated);



%% Set return value and update state
pose = H_WC; 

state.poses = [poses, H_WC(:)];

state.landmark_keypoints = tracked_landmarks_frame_keypoints;
state.landmark_descriptors = tracked_landmarks_frame_descriptors;

if use_KLT
    state.landmark_transforms = tracked_landmarks_transforms;
else
    state.landmark_transforms = [];
end

state.landmarks = tracked_landmarks;
state.candidate_keypoints = [tracked_candidate_keypoints, frame_keypoints];
state.candidate_descriptors = [tracked_candidate_descriptors, frame_descriptors];
state.candidate_bearings_1 = [candidate_bearings_first, frame_bearings];
state.candidate_keypoints_1 = [candidate_keypoints_first, frame_keypoints];
state.candidate_pose_idx_1 = [candidate_pose_idx_first, N_frames*ones(1,size(frame_keypoints,2))];

%% Assertions - not including transforms since these depend on use of KLT or not
num_landmarks = size(state.landmark_keypoints,2);
assert(size(state.landmark_keypoints,2)==num_landmarks);
assert(size(state.landmark_descriptors,2)==num_landmarks);
assert(size(state.landmarks,2)==num_landmarks);

num_candidate_keypoints = size(state.candidate_keypoints,2);
assert(size(state.candidate_keypoints,2)==num_candidate_keypoints);
assert(size(state.candidate_descriptors,2)==num_candidate_keypoints);
assert(size(state.candidate_bearings_1,2)==num_candidate_keypoints);
assert(size(state.candidate_keypoints_1,2)==num_candidate_keypoints);
assert(size(state.candidate_pose_idx_1,2)==num_candidate_keypoints);

assert(size(state.poses,2)==N_frames);

end