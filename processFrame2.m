function [pose, state] = processFrame2(img, K, H_W_prev, oldState)
        
%% Dependencies
addpath('continuous_dependencies/all_solns/00_camera_projection');
addpath('continuous_dependencies/all_solns/01_pnp');
addpath('continuous_dependencies/all_solns/02_detect_describe_match');
addpath('continuous_dependencies/all_solns/04_8point', 'continuous_dependencies/all_solns/04_8point/triangulation', 'continuous_dependencies/all_solns/04_8point/8point');
addpath('continuous_dependencies/all_solns/05_ransac');
addpath('continuous_dependencies/all_solns/07_LK_Tracker');
addpath('continuous_dependencies/');

triangulationAngleThresh = 1*pi/180;
triangulationCosThresh = cos(triangulationAngleThresh);
pixel_tolerance_localization = 10;

%% Extract variables from previous state
poses = oldState.poses;
N_frames = size(poses,2) + 1;

landmarks = oldState.landmarks;
landmark_keypoints = oldState.landmark_keypoints;
landmark_descriptors = oldState.landmark_descriptors;
num_landmarks = size(landmark_keypoints,2);

candidate_keypoints = oldState.candidate_keypoints;
candidate_descriptors = oldState.candidate_descriptors;
candidate_bearing_1 = oldState.candidate_bearings_1;
candidate_keypoints_1 = oldState.candidate_keypoints_1;
candidate_T_idx_1 = oldState.candidate_pose_idx_1;
num_candidates = size(candidate_keypoints, 2);

%% Size test start %%
num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);
assert(size(candidate_bearing_1,2)==num_candidate_keypoints);
assert(size(candidate_keypoints_1,2)==num_candidate_keypoints);
assert(size(candidate_T_idx_1,2)==num_candidate_keypoints);
%% Size test end %%

%% Get keypoints and descriptors in current image
[img_keypoints, img_descriptors] = getHarrisFeatures(img);

%% Match landmark keypoints and candidate keypoints from previous images with current image
match_lambda = 5;
pixel_distance_limit = 200;

% Stack all keypoints which we want to match
prev_keypoints = [landmark_keypoints, candidate_keypoints_1];
prev_descriptors = [landmark_descriptors, candidate_descriptors];
has_match = zeros(1,num_landmarks + num_candidates)';
corresponding_keypoints = zeros(1,num_landmarks + num_candidates)';

% Use match harris descriptors
% This returns a 1xQ matrix where the i-th coefficient is the index of the
% database descriptor which matches to the i-th query descriptor.
% query = img_ (i.e. current)
% database = prev_ (i.e. old)
% (p1,p0) NOT as in init
matches = matchDescriptorsLocally(img_descriptors,prev_descriptors,img_keypoints,prev_keypoints,match_lambda,pixel_distance_limit);

[~, idx_matched_img, idx_matched_prev] = find(matches);
%[~, idx_img_no_match, ~] = find(matches==0);

fprintf('Number of keypoints matched: %i\n',nnz(idx_matched_img));

figure(5);
imshow(img); hold on;
plotMatches(matches, flipud(img_keypoints), flipud(prev_keypoints));
title('Tracked keypoints')
hold off;
pause(0.01);    

has_match(idx_matched_prev) = 1;
corresponding_keypoints(idx_matched_prev) = idx_matched_img;

has_match_landmarks = has_match(1:num_landmarks)>0;
has_match_candidate_keypoints = has_match(num_landmarks+1:end) > 0;

% Index of the updated keypoints (kp's in current image) for keypoints where tracking was successfull
corresponding_candidate_keypoints_idx = corresponding_keypoints(num_landmarks+1:end);
corresponding_candidate_keypoints_idx = corresponding_candidate_keypoints_idx(corresponding_candidate_keypoints_idx>0);

if false
   disp('--Harris matching debug--')
   disp(nnz(has_match(1:num_landmarks))) 
   disp(nnz(has_match(num_landmarks+1:end)));
   disp(all(img_keypoints(matches == 0) == img_keypoints(idx_img_no_match)))
end

% Replace newest candidate keypoint observation with the one from the
% current image. Important to keep ordering here.
candidate_keypoints(:,has_match_candidate_keypoints) = img_keypoints(:,corresponding_candidate_keypoints_idx);

% Keep only landmarks we were able to track
landmark_keypoints = landmark_keypoints(:,has_match_landmarks);
landmark_descriptors = landmark_descriptors(:,has_match_landmarks);
landmarks = landmarks(:,has_match_landmarks);

fprintf('Number of landmarks matched: %i\n',nnz(has_match_landmarks));

% Keep only keypoints we were able to track
candidate_keypoints = candidate_keypoints(:,has_match_candidate_keypoints);
candidate_descriptors = candidate_descriptors(:,has_match_candidate_keypoints);
candidate_bearing_1 = candidate_bearing_1(:,has_match_candidate_keypoints);
candidate_keypoints_1 = candidate_keypoints_1(:,has_match_candidate_keypoints);
candidate_T_idx_1 = candidate_T_idx_1(:,has_match_candidate_keypoints);

%% Size test start %%
num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);
assert(size(candidate_bearing_1,2)==num_candidate_keypoints);
assert(size(candidate_keypoints_1,2)==num_candidate_keypoints);
assert(size(candidate_T_idx_1,2)==num_candidate_keypoints);
%% Size test end %%

%% Estimate relative pose

[H_C_W, inliers] = ransacLocalization(landmark_keypoints, landmarks, K,pixel_tolerance_localization);
fprintf('Number of inliers found: %i\n',sum(inliers));

% Remove outlier landmark-keypoint-correspondences
landmark_keypoints = landmark_keypoints(:,inliers);
landmark_descriptors = landmark_descriptors(:,inliers);
landmarks = landmarks(:,inliers);

% Compute homogenous transform
H_W_C = H_C_W\eye(4);     % frame World to 1

% New candidate keypoint set from current image - updated here since transform
% from RANSASC is needed in bearing vector calculation. Important: These
% are candidate keypoints where the track starts in the current image. I.e.
% NOT candidate keypoints tracked from previous images
new_candidate_keypoints = img_keypoints(:,matches == 0);
new_candidate_descriptors = img_descriptors(:,matches == 0);
new_candidate_bearing_vectors = calculateBearingVectors(new_candidate_keypoints,H_W_C,K);

%% Size test start %%
num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);
assert(size(candidate_bearing_1,2)==num_candidate_keypoints);
assert(size(candidate_keypoints_1,2)==num_candidate_keypoints);
assert(size(candidate_T_idx_1,2)==num_candidate_keypoints);
%% Size test end %%

%% Triangulate new Landmarks

% Calculate bearing vector for observation in current image
candidate_bearing_2 = calculateBearingVectors(candidate_keypoints,H_W_C,K);

% Triangulability test. The angle between the bearing vectors must be large
% enough. innerProd = cos(angle)... ==> low innerProd = large angle
can_triangulate = find(sum(candidate_bearing_1.*candidate_bearing_2,1) < 1);

p1 = candidate_keypoints_1(:,can_triangulate);
p1_poses_ind = candidate_T_idx_1(can_triangulate);
p2 = candidate_keypoints(:,can_triangulate);

p1 = homogenize2D(p1);
p2 = homogenize2D(p2);

assert(all(size(p1)==size(p2)));

M2 = K*H_C_W(1:3,:); % Projection matrix for current image (image 2)

% Triangulate landmarks one by one, since they may have different poses
candidate_landmarks_W = zeros(4,size(p2,2));

for i= 1:size(p2,2)
    % Projection matrix for image 1, M1, varies for all keypoints
    H_iW = reshape(poses(:,p1_poses_ind(i)),4,4)\eye(4);
    M1 = K*H_iW(1:3,:);
    candidate_landmarks_W(:,i) = linearTriangulation(p1(:,i), p2(:,i), M1, M2);
end

% Find triangulated points in front of camera and closer than a threshold
candidate_landmarks_C = H_C_W*candidate_landmarks_W;
posZInds = (candidate_landmarks_C(3,:) > 0 & sqrt(sum(candidate_landmarks_C(1:3,:).^2,1)) < 6000);
new_landmarks = candidate_landmarks_W(:,posZInds);
did_triangulate = can_triangulate(posZInds);

fprintf('Num landmarks triangulated: %i\n',nnz(did_triangulate));

% Update landmarks
landmarks = [landmarks, new_landmarks];
landmark_keypoints = [landmark_keypoints, candidate_keypoints(:,did_triangulate)];
landmark_descriptors = [landmark_descriptors, candidate_descriptors(:,did_triangulate)];

% Remove new landmark keypoints from candidate keypoints
candidate_keypoints(:,did_triangulate) = [];
candidate_descriptors(:,did_triangulate) = [];
candidate_bearing_1(:,did_triangulate) = [];
candidate_keypoints_1(:,did_triangulate) = [];
candidate_T_idx_1(:,did_triangulate) = [];

%% Update candidates with keypoints from current image
candidate_keypoints = [candidate_keypoints new_candidate_keypoints];
candidate_descriptors = [candidate_descriptors new_candidate_descriptors];
candidate_bearing_1 = [candidate_bearing_1 new_candidate_bearing_vectors];
candidate_keypoints_1 = [candidate_keypoints_1 new_candidate_keypoints];
candidate_T_idx_1 = [candidate_T_idx_1 N_frames*ones(1,size(new_candidate_keypoints,2))];

%% Update poses
pose = H_W_C; 
poses = [poses pose(:)];

%% Set new state
state.poses = poses;
state.landmark_keypoints = landmark_keypoints;
state.landmark_descriptors = landmark_descriptors;
state.landmark_transforms = [];
state.landmarks = landmarks;

state.candidate_keypoints = candidate_keypoints;
state.candidate_descriptors = candidate_descriptors;
state.candidate_bearings_1 = candidate_bearing_1;
state.candidate_keypoints_1 = candidate_keypoints_1;
state.candidate_pose_idx_1 = candidate_T_idx_1;

%% Assertions
num_landmarks = size(landmark_keypoints,2);
assert(size(landmark_keypoints,2)==num_landmarks);
assert(size(landmark_descriptors,2)==num_landmarks);
assert(size(landmarks,2)==num_landmarks);

num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);
assert(size(candidate_bearing_1,2)==num_candidate_keypoints);
assert(size(candidate_keypoints_1,2)==num_candidate_keypoints);
assert(size(candidate_T_idx_1,2)==num_candidate_keypoints);

assert(size(poses,2)==N_frames);

end