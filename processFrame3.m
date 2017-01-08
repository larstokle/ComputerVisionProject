function [pose, state] = processFrame3(img, K, H_W_prev, oldState)
        
%% Dependencies
addpath('init_dependencies/8point/');
addpath('continuous_dependencies/all_solns/00_camera_projection');
addpath('continuous_dependencies/all_solns/01_pnp');
addpath('continuous_dependencies/all_solns/02_detect_describe_match');
addpath('continuous_dependencies/all_solns/04_8point', 'continuous_dependencies/all_solns/04_8point/triangulation', 'continuous_dependencies/all_solns/04_8point/8point');
addpath('continuous_dependencies/all_solns/05_ransac');
addpath('continuous_dependencies/all_solns/07_LK_Tracker');
addpath('continuous_dependencies/');

%% Options
do_plot = false;

%% Extract variables from previous state
poses = oldState.poses;
N_frames = size(poses,2) + 1;

landmark_keypoints = oldState.landmark_keypoints;
landmark_descriptors = oldState.landmark_descriptors;
first_obs = oldState.first_obs; % Frame of the first observation of the landmark
num_landmarks = size(landmark_keypoints,2);

candidate_keypoints = oldState.candidate_keypoints;
candidate_descriptors = oldState.candidate_descriptors;
num_candidates = size(candidate_keypoints, 2);

%% Size test start %%
num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);
%% Size test end %%

%% Get keypoints and descriptors in current image
[img_keypoints, img_descriptors] = getHarrisFeatures(img);

%% Match landmark keypoints and candidate keypoints from previous images with current image
match_lambda = 5;
pixel_distance_limit = 200;

% Stack all keypoints which we want to match
prev_keypoints = [landmark_keypoints, candidate_keypoints];
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
%[~, idx_img_no_match,idx_prev_no_match] = find(matches==0);

fprintf('Number of keypoints matched: %i\n',nnz(idx_matched_img));  

has_match(idx_matched_prev) = 1;
corresponding_keypoints(idx_matched_prev) = idx_matched_img;

has_match_landmarks = has_match(1:num_landmarks)>0;
has_match_candidate_keypoints = has_match(num_landmarks+1:end) > 0;

% Index of the updated keypoints (kp's in current image) for keypoints where tracking was successfull
corresponding_candidate_keypoints_idx = corresponding_keypoints(num_landmarks+1:end);
corresponding_candidate_keypoints_idx = corresponding_candidate_keypoints_idx(corresponding_candidate_keypoints_idx>0);

corresponding_landmark_keypoint_idx = corresponding_keypoints(1:num_landmarks);
corresponding_landmark_keypoint_idx = corresponding_landmark_keypoint_idx(corresponding_landmark_keypoint_idx>0);

if false
   disp('--Harris matching debug--')
   disp(nnz(has_match(1:num_landmarks))) 
   disp(nnz(has_match(num_landmarks+1:end)));
   disp(all(img_keypoints(matches == 0) == img_keypoints(idx_img_no_match)))
end

% Replace newest candidate keypoint observation with the one from the
% current image. Important to keep ordering here.
candidate_keypoints_prev =  candidate_keypoints(:,has_match_candidate_keypoints);
candidate_keypoints(:,has_match_candidate_keypoints) = img_keypoints(:,corresponding_candidate_keypoints_idx);
candidate_descriptors(:,has_match_candidate_keypoints) = img_descriptors(:,corresponding_candidate_keypoints_idx);

% Keep only landmarks we were able to track
landmark_keypoints_prev = landmark_keypoints(:,has_match_landmarks);

landmark_keypoints = img_keypoints(:,corresponding_landmark_keypoint_idx);
landmark_descriptors = img_descriptors(:,corresponding_landmark_keypoint_idx);
first_obs = first_obs(:,has_match_landmarks);

fprintf('Number of landmarks matched: %i\n',nnz(has_match_landmarks));

% Keep only keypoints we were able to track
candidate_keypoints = candidate_keypoints(:,has_match_candidate_keypoints);
candidate_descriptors = candidate_descriptors(:,has_match_candidate_keypoints);
candidate_first_obs = (N_frames-1)*ones(1,size(candidate_keypoints,2));

%% Size test start %%
num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);
assert(size(candidate_first_obs,2)==num_candidate_keypoints);
%% Size test end %%

%% Augment landmarks
landmark_keypoints_prev = [landmark_keypoints_prev candidate_keypoints_prev];
landmark_keypoints = [landmark_keypoints candidate_keypoints];
landmark_descriptors = [landmark_descriptors candidate_descriptors];
first_obs = [first_obs candidate_first_obs];

%% Estimate pose

% 1 point ransac here
[theta_est,~,inliers,H_C1_prev] = estimateTheta(landmark_keypoints_prev,landmark_keypoints,K,1);

disp(['Estimated theta: ' num2str(rad2deg(theta_est))]);
disp(['Num inliers 1p: ' num2str(nnz(inliers)) ]);

% Relative pose via fundamental matrix estimation
num_inliers = nnz(inliers);

if num_inliers >= 20 && abs(rad2deg(theta_est)) <= 10
    %% Estimate essential matrix with normalized 8-p-algorithm
    p1_hom = homogenize2D(landmark_keypoints_prev(:,inliers));
    p2_hom = homogenize2D(landmark_keypoints(:,inliers));
    
    % User normalized 8-p-a to estimate essential matrix
    E = estimateEssentialMatrix(p1_hom,p2_hom,K,K);
    
    % Obtain extrinsic parameters (R,t) from E
    [Rots,u3] = decomposeEssentialMatrix(E);
    
    % Disambiguate among the four possible configurations
    [R_C1_prev,T_C1_prev] = disambiguateRelativePose(Rots,u3,p1_hom,p2_hom,K,K);
    H_C1_prev = [R_C1_prev , T_C1_prev ; 0 0 0 1];    
    
elseif num_inliers >= 8
    disp('Starting 8 point fundamental matrix estimation')
    beep
    
    %% Check if we find more inliers using 8-p-ransac
    p1_hom = homogenize2D(landmark_keypoints_prev);
    p2_hom = homogenize2D(landmark_keypoints);
    
    epsilon = num_inliers / numel(inliers);
    p = 0.90;
    num_iter_8p_RANSAC = log(1-p)/log(1-epsilon^8);
    num_iter_8p_RANSAC = min([num_iter_8p_RANSAC 10000]);
    num_iter_8p_RANSAC = max([num_iter_8p_RANSAC 1000]);
    
    [E,eightp_inlier_mask] = estimateEssentialMatrix_RANSAC(p1_hom,p2_hom,K,K,num_iter_8p_RANSAC);
    
    disp('Terminating 8 point fundamental matrix estimation')
    
    if nnz(eightp_inlier_mask) > 0
        % Obtain extrinsic parameters (R,t) from E
        [Rots,u3] = decomposeEssentialMatrix(E);

        % Disambiguate among the four possible configurations
        [R_C1_prev,T_C1_prev] = disambiguateRelativePose(Rots,u3,p1_hom,p2_hom,K,K);
        H_C1_prev = [R_C1_prev , T_C1_prev ; 0 0 0 1];           
        
        inliers = eightp_inlier_mask;
        disp(['Num inliers 8p: ' num2str(nnz(inliers)) ]);
    end
    
elseif num_inliers == 1
    % 1 point - i.e. keep what is goin on
else
    % Run init again or other fallback. Such as using previous estimate.
end


if do_plot

    figure(11);
    subplot(1,3,[1 2]);
    imshow(img); hold on;
    plotMatchVectors(landmark_keypoints_prev(:,inliers==0),landmark_keypoints(:,inliers==0),'r');
    plotMatchVectors(landmark_keypoints_prev(:,inliers),landmark_keypoints(:,inliers),'g');
    title(['Tracked points. Num inliers=' num2str(nnz(inliers)) ' Num outliers=' num2str(nnz(inliers==0))]);

    subplot(1,3,3);
    frames_in = sort(unique(first_obs(inliers)));
    freq_in = sum(first_obs(inliers) == frames_in',2);
    plot(frames_in,freq_in,'g-');
    hold on
    frames_out = sort(unique(first_obs(inliers==0)));
    freq_out = sum(first_obs(inliers==0) == frames_out',2);
    plot(frames_out,freq_out,'r-');

    % Guard. If min==max matlab complains
    if (min([frames_in,frames_out])-max([frames_in,frames_out]))>0
        xlim([min([frames_in,frames_out]) max([frames_in,frames_out])]);
    end

    % Guard. If 0==max matlab complains
    if max([freq_in(:);freq_out(:)]')>0
        ylim([0 max([freq_in(:);freq_out(:)]')])
    end

    assert(nnz(inliers)==sum(freq_in));
    assert(nnz(inliers==0)==sum(freq_out));
    hold off;
    drawnow;

end

%% Candidate keypoints from this frame
% Note: Outlier landmark keypoints can still be re-used as keypoints from
% this frame
new_candidate_keypoints = [img_keypoints(:,matches == 0) landmark_keypoints(:,inliers==0)];
new_candidate_descriptors = [img_descriptors(:,matches == 0) landmark_descriptors(:,inliers==0)];

% Remove outliers
landmark_keypoints = landmark_keypoints(:,inliers);
landmark_keypoints_prev = landmark_keypoints_prev(:,inliers);
landmark_descriptors = landmark_descriptors(:,inliers);
first_obs = first_obs(:,inliers);

%% Size test start %%
num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);
%% Size test end %%

%% Update candidates with keypoints from current image
candidate_keypoints = new_candidate_keypoints;
candidate_descriptors = new_candidate_descriptors;

%% Update poses
H_W_prev = reshape(poses(:,end),4,4);
H_prev_W = H_W_prev^-1;
H_C1_W = H_C1_prev*H_prev_W;

pose = H_C1_W^-1; % Pose from 1-p-ransac
poses = [poses pose(:)];

%% Set new state
state.poses = poses;
state.landmark_keypoints = landmark_keypoints;
state.landmark_descriptors = landmark_descriptors;
state.first_obs = first_obs;

state.candidate_keypoints = candidate_keypoints;
state.candidate_descriptors = candidate_descriptors;

%% Assertions
num_landmarks = size(landmark_keypoints,2);
assert(size(landmark_keypoints,2)==num_landmarks);
assert(size(landmark_descriptors,2)==num_landmarks);
assert(size(first_obs,2)==num_landmarks);

num_candidate_keypoints = size(candidate_keypoints,2);
assert(size(candidate_keypoints,2)==num_candidate_keypoints);
assert(size(candidate_descriptors,2)==num_candidate_keypoints);


assert(size(poses,2)==N_frames);

end