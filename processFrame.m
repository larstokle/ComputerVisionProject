function [pose, state] = processFrame(img, K, H_W0, oldState)
%[pose, state] = processFrame(img, K, H_W0, oldState):
%
%Tracks keypoints from last image and estimates relative pose.
%Finds new keypoints and updates tracker.
%If keypoints are "good enough", are turned into landmarks
%   
%   Input:
%       img:            image frame to process
%       K:              intrinsic matrix of camera
%       H_W0:           homogenous transformation to where the
%           previous frame was
%       oldState: struct...
%       {
%       tracker:   a vision.PointTracker object that has processed the
%           previous frame
%       keypoints:      (2xN) list of keypoints in image coord tracked by
%           pointTRacker
%       landmarks:  (4xN) list of homogenous landmarks
%       }
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

%% init
mapAx = get(2,'CurrentAxes');
estAx = get(3,'CurrentAxes');
potAx = get(4,'CurrentAxes');

% tuning parameters new keypoints
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1500;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;

%tuning parameters established landmarks and pose estimation
use_KLT = true;
KLT_match = 0.001; %fraction of maximum patch distance
match_lambda_est = 12;
pixel_motion_err_tol = 75;
reprojection_pix_tol = 10;

%tuning parameters potential landmarks
match_lambda_pot = 7;
max_epipole_line_dist = 15;
max_match_dist = 150;
triangulationAngleThresh = 2*pi/180;
triangulationCosThresh = cos(triangulationAngleThresh);



%extract variables from state
poses = oldState.poses;
N_frames = size(poses,2) + 1;

P_landmarks_W = oldState.landmarks;

landmark_keypoints = oldState.landmark_keypoints;
landmark_descriptors = oldState.landmark_descriptors;
landmark_transforms = oldState.landmark_transforms;
N_landmarks = size(landmark_keypoints,2);
patch_radius = sqrt(size(landmark_descriptors,1));

candidate_keypoints = oldState.candidate_keypoints;
candidate_descriptors_1 = oldState.candidate_descriptors_1;
candidate_bearings_1 = oldState.candidate_bearings_1;
candidate_keypoints_1 = oldState.candidate_keypoints_1;
candidate_pose_idx_1 = oldState.candidate_pose_idx_1;
N_candidates = size(candidate_keypoints, 2);

%% find points in this frame to query
harrisScore = harris(img, harris_patch_size, harris_kappa);
frame_keypoints = selectKeypoints(harrisScore, num_keypoints, nonmaximum_supression_radius);
frame_descriptors = describeKeypoints(img, frame_keypoints, descriptor_radius);
frame_keypoints = flipud(frame_keypoints);

%% track established points

%estimate new point with same homogenous transform as last transform
H_W_lastlast = reshape(poses(:,end-1),4,4);
H_W_last = reshape(poses(:,end),4,4);

%translation estimation
w_t_lastlast_last = H_W_last(1:3,4) - H_W_lastlast(1:3,4); %last translation
w_t_last_C_est = [w_t_lastlast_last(1); 0; w_t_lastlast_last(3)]; % direction like last translation in z and x only
t_last_C_est = H_W_lastlast(1:3,1:3)'*w_t_last_C_est*norm(w_t_lastlast_last)/norm(w_t_last_C_est); % magnitude like last tranlation

%rotation estimation
R_lastlast_last = H_W_lastlast(1:3,1:3)'*H_W_last(1:3,1:3); %last rotation
omega_hat_lastlast_last = logm(R_lastlast_last); %last skew
theta_lastlast_last = norm(matrix2cross(omega_hat_lastlast_last)); %last angle of rotation
omega_last_C_est = H_W_lastlast(1:3,1:3)'*[0; theta_lastlast_last; 0]; %rotation magnitude equal to last but limited to around y axis
R_last_C_est = expm(cross2matrix(omega_last_C_est));

%homogenouos transform estimate
H_last_C_est = [R_last_C_est, t_last_C_est;
                0, 0, 0, 1];
H_W_C_est = H_W_last*H_last_C_est;

%keypoint localization estimate
P_landmarks_C_est = H_W_C_est\P_landmarks_W;

if use_KLT
    landmark_transforms(5:6,:) = projectPoints(P_landmarks_C_est(1:3,:), K);
else
    landmark_position_estimate = projectPoints(P_landmarks_C_est(1:3,:), K);
end

% tracking
if use_KLT
    % track with pyramid KLT
    establishedDescriptorsImg = reshape(landmark_descriptors,[patch_radius, patch_radius, N_landmarks]);
	[ps, ds, valid_KLT] = KLTtracker(img, landmark_transforms, establishedDescriptorsImg, KLT_match);
else
    matches = matchDescriptorsEpiPolar(landmark_descriptors, frame_descriptors, landmark_position_estimate, frame_keypoints, match_lambda_est,[],[],0,pixel_motion_err_tol);
end

%discard nonvalid tracks
if use_KLT
    tracked_landmark_keypoints = ds(:,valid_KLT);
    tracked_landmark_descriptors = landmark_descriptors(:,valid_KLT); %old dexcriptors used
    %trackedEstablishedDescriptors = describeKeypoints(img,flipud(round(trackedEstablishedKeypoints)), descriptor_radius); % if new descriptor is wanted
    tracked_landmark_transforms = ps(:,valid_KLT);
    
    tracked_landmarks = P_landmarks_W(:,valid_KLT);
    establishedKeypointsFound = landmark_keypoints(:,valid_KLT);
    
    [~, isNew] = setdiff(...
                round(frame_keypoints'/nonmaximum_supression_radius),...
                round(tracked_landmark_keypoints'/nonmaximum_supression_radius),...
                'rows');
    frame_keypoints = frame_keypoints(:,isNew);
    frame_descriptors = frame_descriptors(:,isNew);
    
else
    [~, establishedInd, newInd] = find(matches);
    
    tracked_landmark_keypoints = frame_keypoints(:,newInd);
    tracked_landmark_descriptors = frame_descriptors(:,newInd);
    
    establishedKeypointsFound = landmark_keypoints(:,establishedInd);
    tracked_landmarks = P_landmarks_W(:,establishedInd);
    
    [~, isNew] = setdiff(...
                round(frame_keypoints'/nonmaximum_supression_radius),...
                round(tracked_landmark_keypoints'/nonmaximum_supression_radius),...
                'rows');
    frame_keypoints = frame_keypoints(:,isNew);
    frame_descriptors = frame_descriptors(:,isNew);
end

%validation plot established points
%set(0,'CurrentFigure',3);clf;
cla(estAx);
imshow(img,'Parent',estAx);
hold(estAx,'on');
scatter(landmark_keypoints(1,:), landmark_keypoints(2,:), 'yx','Linewidth',2','Parent',estAx);
plot([establishedKeypointsFound(1,:); tracked_landmark_keypoints(1,:)],[establishedKeypointsFound(2,:); tracked_landmark_keypoints(2,:)],'r','Linewidth',1,'Parent',estAx);
title('tracked established keypoints from last frame','Parent',estAx);
scatter(tracked_landmark_keypoints(1,:), tracked_landmark_keypoints(2,:),'rx','Linewidth',2,'Parent',estAx);

landmark_keypoints = establishedKeypointsFound; %done here so that the old established can be plotted

%% Estimate relative pose

[H_CW, inliers] = ransacLocalization(tracked_landmark_keypoints, tracked_landmarks, K, reprojection_pix_tol);
fprintf('RANSAC: #inliers: %i, #outliers: %i\n',sum(inliers),sum(inliers==0));

tracked_landmark_keypoints = tracked_landmark_keypoints(:,inliers);
tracked_landmark_descriptors = tracked_landmark_descriptors(:,inliers);
if use_KLT
    tracked_landmark_transforms = tracked_landmark_transforms(:,inliers);
end

tracked_landmarks = tracked_landmarks(:,inliers);
landmark_keypoints = landmark_keypoints(:,inliers);

%validate ransac
plot([landmark_keypoints(1,:); tracked_landmark_keypoints(1,:)],[landmark_keypoints(2,:); tracked_landmark_keypoints(2,:)],'g','Linewidth',1,'Parent',estAx);

%% compute homogenous transforms
H_WC = H_CW\eye(4);     % frame World to 1
H_last_C = H_W_last\H_WC;


%% track potential points
matches = matchDescriptorsEpiPolar(candidate_descriptors_1, frame_descriptors, candidate_keypoints, frame_keypoints, match_lambda_pot, H_last_C, K, max_epipole_line_dist, max_match_dist);

[~, potentialInd, newInd] = find(matches);
tracked_candidate_keypoints = frame_keypoints(:,newInd);
tracked_candidate_descriptors = frame_descriptors(:,newInd);
N_candidates_tracked = size(tracked_candidate_keypoints,2);

%remove tracked potential points from new
[~, newIndNon] = setdiff(frame_keypoints',tracked_candidate_keypoints','rows');
frame_keypoints = frame_keypoints(:,newIndNon);
frame_descriptors = frame_descriptors(:,newIndNon);

%calculate new bearings
frame_bearings = K\[frame_keypoints; ones(1,size(frame_keypoints,2))];
frame_bearings = H_WC(1:3,1:3)*(frame_bearings./(ones(3,1)*sqrt(sum(frame_bearings.^2,1))));

%validation plot potentials
%set(0,'CurrentFigure',4);clf;
cla(potAx);
imshow(img,'Parent',potAx);
hold(potAx,'on');
scatter(candidate_keypoints_1(1,:), candidate_keypoints_1(2,:), 'yx','Linewidth',2,'Parent',potAx);
plot([candidate_keypoints_1(1,potentialInd); tracked_candidate_keypoints(1,:)],[candidate_keypoints_1(2,potentialInd); tracked_candidate_keypoints(2,:)],'g','Linewidth',1, 'Parent',potAx);
scatter(tracked_candidate_keypoints(1,:), tracked_candidate_keypoints(2,:),'rx','Linewidth',2, 'Parent',potAx);

% remove nontracked points after ploting
candidate_bearings_1 = candidate_bearings_1(:,potentialInd);
candidate_keypoints_1 = candidate_keypoints_1(:,potentialInd);
candidate_pose_idx_1 = candidate_pose_idx_1(:,potentialInd);

%% triangulate new landmarks

%calculate new bearing
trackedPotentialKeypointsHom = [tracked_candidate_keypoints; ones(1,size(tracked_candidate_keypoints,2))];
trackedPotentialKeypointsNorm = K\trackedPotentialKeypointsHom;
trackedLens = sqrt(sum(trackedPotentialKeypointsNorm.^2,1));
trackedPotentialKeypointsNormBearing = H_WC(1:3,1:3)*(trackedPotentialKeypointsNorm./(ones(3,1)*trackedLens));

% innerprod = cos(angle)... ==> low innerProd = large angle
innerProd = sum(trackedPotentialKeypointsNormBearing.*candidate_bearings_1,1);
goodPotentialInds = find(innerProd < triangulationCosThresh);

p1_bearings = candidate_bearings_1(:,goodPotentialInds);
p1_poses_ind = candidate_pose_idx_1(goodPotentialInds);
p2 = trackedPotentialKeypointsHom(:,goodPotentialInds);

%triangulate landmarks one by one, since they may have different pose
newLandmarks = zeros(4,size(p2,2));
for i= 1:size(p2,2)
    H_iW = reshape(poses(:,p1_poses_ind(i)),4,4)\eye(4);
    p1 = K*H_iW(1:3,1:3)*p1_bearings(:,i);
    p1 = p1(1:3)/p1(3);
    newLandmarks(:,i) = linearTriangulation(p1, p2(:,i), K*H_iW(1:3,:), K*H_CW(1:3,:));
end

%find those that are in front of camera and closer than a threshold
newLandmarks_C = H_CW*newLandmarks;
posZInds = find(newLandmarks_C(3,:) > 0 & sqrt(sum(newLandmarks_C(1:3,:).^2,1)) < 60);
newLandmarks = newLandmarks(:,posZInds);
goodPotentialInds = goodPotentialInds(posZInds);

% ============ check this thing now!!! working? >

%update established landmarks
tracked_landmarks = [tracked_landmarks, newLandmarks];
tracked_landmark_keypoints = [tracked_landmark_keypoints, tracked_candidate_keypoints(:,goodPotentialInds)];
tracked_landmark_descriptors = [tracked_landmark_descriptors, tracked_candidate_descriptors(:,goodPotentialInds)];
if use_KLT
    tracked_landmark_transforms = [tracked_landmark_transforms,...
        [zeros(4,size(tracked_candidate_keypoints(:,goodPotentialInds),2));tracked_candidate_keypoints(:,goodPotentialInds)]];
end
    
stillPotentialInds = find(innerProd >= triangulationCosThresh);
tracked_candidate_keypoints = tracked_candidate_keypoints(:,stillPotentialInds);
tracked_candidate_descriptors = tracked_candidate_descriptors(:,stillPotentialInds);

candidate_bearings_1 = candidate_bearings_1(:, stillPotentialInds);
candidate_keypoints_1 = candidate_keypoints_1(:, stillPotentialInds);
candidate_pose_idx_1 = candidate_pose_idx_1(stillPotentialInds);
% ============<

%validate plot new landmarks
%set(0,'CurrentFigure',4);
scatter(p2(1,:),p2(2,:),'g','Linewidth',2, 'Parent', potAx);

%set(0,'CurrentFigure',2);
scatter(newLandmarks(3,:), -newLandmarks(1,:),'.r', 'Parent', mapAx);
title(['tracked potential keypoints from last frame, #tracked = ',num2str(N_candidates_tracked), ', #triangulated = ', num2str(size(newLandmarks,2))],'Parent',potAx);
axis(mapAx,'equal');

%% return values
pose = H_WC; 

state.poses = [poses, H_WC(:)];


state.landmark_keypoints = tracked_landmark_keypoints;
state.landmark_descriptors = tracked_landmark_descriptors;

if use_KLT
    state.landmark_transforms = tracked_landmark_transforms;
else
    state.landmark_transforms = [];
end

state.landmarks = tracked_landmarks;
state.candidate_keypoints = [tracked_candidate_keypoints, frame_keypoints];
state.candidate_descriptors_1 = [tracked_candidate_descriptors, frame_descriptors];
state.candidate_bearings_1 = [candidate_bearings_1, frame_bearings];
state.candidate_keypoints_1 = [candidate_keypoints_1, frame_keypoints];
state.candidate_pose_idx_1 = [candidate_pose_idx_1, N_frames*ones(1,size(frame_keypoints,2))];

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