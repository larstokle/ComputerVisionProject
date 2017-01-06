function [pose, state] = processFrame(img, K, H_W0, old_state)
%[pose, state] = processFrame(img, K, H_W0, oldState):
%   
%   Input:
%       img:            frame to process
%       K:              camera calibration matrix
%       H_W0:           homogenous transformation to where the
%       old_state:       struct
%
%   Output:
%       pose:   pose computed at img relative to world frame
%       state:  updated input struct
%           

%% Dependencies
addpath('continuous_dependencies/all_solns/00_camera_projection');
addpath('continuous_dependencies/all_solns/01_pnp');
addpath('continuous_dependencies/all_solns/02_detect_describe_match');
addpath('continuous_dependencies/all_solns/04_8point', 'continuous_dependencies/all_solns/04_8point/triangulation', 'continuous_dependencies/all_solns/04_8point/8point');
addpath('continuous_dependencies/all_solns/05_ransac');
addpath('continuous_dependencies/all_solns/07_LK_Tracker');
addpath('continuous_dependencies/');

%% Init plotting
mapAx = get(2,'CurrentAxes');
estAx = get(3,'CurrentAxes');
potAx = get(4,'CurrentAxes');

% Tuning parameters frame (new) keypoints
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1500;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;

% Tuning parameters landmarks and pose estimation
use_KLT = true;
KLT_match = 0.001; %fraction of maximum patch distance
match_lambda_est = 12;
pixel_motion_err_tol = 75;
reprojection_pix_tol = 10;

% Tuning parameters candidate keypoints and triangulation
match_lambda_candidates = 7;
max_epipole_line_dist = 15;
max_match_dist = 150;
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
candidate_descriptors_first = old_state.candidate_descriptors_1;
candidate_bearings_first = old_state.candidate_bearings_1;
candidate_keypoints_first = old_state.candidate_keypoints_1;
candidate_pose_idx_first = old_state.candidate_pose_idx_1;
N_candidates = size(candidate_prev_keypoints, 2);

%% find points in this frame to query
harris_score = harris(img, harris_patch_size, harris_kappa);
frame_keypoints = selectKeypoints(harris_score, num_keypoints, nonmaximum_supression_radius);
frame_descriptors = describeKeypoints(img, frame_keypoints, descriptor_radius);
frame_keypoints = flipud(frame_keypoints);

%% track established points

%estimate new point with same homogenous transform as last transform
H_W_lastlast = reshape(poses(:,end-1),4,4);
H_W_last = reshape(poses(:,end),4,4);

% Estimate translation
w_t_lastlast_last = H_W_last(1:3,4) - H_W_lastlast(1:3,4); %last translation
w_t_last_C_est = [w_t_lastlast_last(1); 0; w_t_lastlast_last(3)]; % direction like last translation in z and x only
t_last_C_est = H_W_lastlast(1:3,1:3)'*w_t_last_C_est*norm(w_t_lastlast_last)/norm(w_t_last_C_est); % magnitude like last tranlation

% Estimate rotation
R_lastlast_last = H_W_lastlast(1:3,1:3)'*H_W_last(1:3,1:3); %last rotation
omega_hat_lastlast_last = logm(R_lastlast_last); %last skew
theta_lastlast_last = norm(matrix2cross(omega_hat_lastlast_last)); %last angle of rotation
omega_last_C_est = H_W_lastlast(1:3,1:3)'*[0; theta_lastlast_last; 0]; %rotation magnitude equal to last but limited to around y axis
R_last_C_est = expm(cross2matrix(omega_last_C_est));

% Estimate homogenous transform
H_last_C_est = [R_last_C_est, t_last_C_est;
                0, 0, 0, 1];
H_W_C_est = H_W_last*H_last_C_est;

% Estimate location of landmark keypoints in the (estimated) current camera frame
landmarks_C = H_W_C_est\landmarks_w;

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
else
    matches = matchDescriptorsEpiPolar(landmark_descriptors, frame_descriptors, landmark_position_estimate, frame_keypoints, match_lambda_est,[],[],0,pixel_motion_err_tol);
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
    [~, establishedInd, current_frame_idx] = find(matches);
    
    tracked_landmarks_frame_keypoints = frame_keypoints(:,current_frame_idx);
    tracked_landmarks_frame_descriptors = frame_descriptors(:,current_frame_idx);
    
    tracked_landmarks_prev_keypoints = landmark_prev_keypoints(:,establishedInd);
    tracked_landmarks = landmarks_w(:,establishedInd);
    
    [~, isNew] = setdiff(...
                round(frame_keypoints'/nonmaximum_supression_radius),...
                round(tracked_landmarks_frame_keypoints'/nonmaximum_supression_radius),...
                'rows');

    frame_keypoints = frame_keypoints(:,isNew);
    frame_descriptors = frame_descriptors(:,isNew);
end

% Validation plot landmark points
%set(0,'CurrentFigure',3);clf;
cla(estAx);
imshow(img,'Parent',estAx);
hold(estAx,'on');
scatter(landmark_prev_keypoints(1,:), landmark_prev_keypoints(2,:), 'yx','Linewidth',2','Parent',estAx);
plot([tracked_landmarks_prev_keypoints(1,:); tracked_landmarks_frame_keypoints(1,:)],[tracked_landmarks_prev_keypoints(2,:); tracked_landmarks_frame_keypoints(2,:)],'r','Linewidth',1,'Parent',estAx);
title('Landmarks tracked from previous frame','Parent',estAx);
scatter(tracked_landmarks_frame_keypoints(1,:), tracked_landmarks_frame_keypoints(2,:),'rx','Linewidth',2,'Parent',estAx);

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
plot([tracked_landmarks_prev_keypoints(1,:); tracked_landmarks_frame_keypoints(1,:)],[tracked_landmarks_prev_keypoints(2,:); tracked_landmarks_frame_keypoints(2,:)],'g','Linewidth',1,'Parent',estAx);

%% Compute homogenous transforms
H_WC = H_CW\eye(4);     % World to 1
H_last_C = H_W_last\H_WC;

%% Track candidate keypoints
matches = matchDescriptorsEpiPolar(candidate_descriptors_first, frame_descriptors, candidate_prev_keypoints, frame_keypoints, match_lambda_candidates, H_last_C, K, max_epipole_line_dist, max_match_dist);

[~, prev_frame_idx, current_frame_idx] = find(matches);
tracked_candidate_keypoints = frame_keypoints(:,current_frame_idx);
tracked_candidate_descriptors = frame_descriptors(:,current_frame_idx);
N_candidates_tracked = size(tracked_candidate_keypoints,2);

%remove tracked candidate keypoints from set of new candidate keypoints -
%old solution here
%[~, newIndNon] = setdiff(frame_keypoints',tracked_candidate_keypoints','rows');
%frame_keypoints = frame_keypoints(:,newIndNon);
%frame_descriptors = frame_descriptors(:,newIndNon);

% New candidate keypoints added at the current frame
frame_keypoints(:,current_frame_idx) = [];
frame_descriptors(:,current_frame_idx) = [];

% Calculate bearing vectors of new candidate keypoints
frame_bearings = calculateBearingVectors(frame_keypoints,H_WC,K);

%validation plot candidates
%set(0,'CurrentFigure',4);clf;
cla(potAx);
imshow(img,'Parent',potAx);
hold(potAx,'on');
scatter(candidate_keypoints_first(1,:), candidate_keypoints_first(2,:), 'yx','Linewidth',2,'Parent',potAx);
plot([candidate_keypoints_first(1,prev_frame_idx); tracked_candidate_keypoints(1,:)],[candidate_keypoints_first(2,prev_frame_idx); tracked_candidate_keypoints(2,:)],'g','Linewidth',1, 'Parent',potAx);
scatter(tracked_candidate_keypoints(1,:), tracked_candidate_keypoints(2,:),'rx','Linewidth',2, 'Parent',potAx);

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

% Update landmarks
tracked_landmarks = [tracked_landmarks, new_landmarks];
tracked_landmarks_frame_keypoints = [tracked_landmarks_frame_keypoints, tracked_candidate_keypoints(:,did_triangulate)];
tracked_landmarks_frame_descriptors = [tracked_landmarks_frame_descriptors, tracked_candidate_descriptors(:,did_triangulate)];

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

% Validation plot new landmarks
%set(0,'CurrentFigure',4);
scatter(p2(1,:),p2(2,:),'g','Linewidth',2, 'Parent', potAx);

%set(0,'CurrentFigure',2);
scatter(new_landmarks(3,:), -new_landmarks(1,:),'.r', 'Parent', mapAx);
title(['Tracked candidate keypoints from last frame, #tracked = ',num2str(N_candidates_tracked), ', #triangulated = ', num2str(size(new_landmarks,2))],'Parent',potAx);
axis(mapAx,'equal');

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
state.candidate_descriptors_1 = [tracked_candidate_descriptors, frame_descriptors];
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
assert(size(state.candidate_descriptors_1,2)==num_candidate_keypoints);
assert(size(state.candidate_bearings_1,2)==num_candidate_keypoints);
assert(size(state.candidate_keypoints_1,2)==num_candidate_keypoints);
assert(size(state.candidate_pose_idx_1,2)==num_candidate_keypoints);

assert(size(state.poses,2)==N_frames);

end