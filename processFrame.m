function [pose, state] = processFrame(img, K, H_W0, oldState)
%initializeVO: finds position of second image and makes a list of landmarks
%   
%   Input:
%       img:            image frame to process
%       K:              intrinsic matrix of camera
%       H_W0:           homogenous transformation to frame where the
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

%% init
numNewFeaturesToFind = 200;
keypoints = oldState.keypoints;
pointTracker = oldState.tracker;
P_landmarks_W = oldState.landmarks;

%% track points
[newKeypoints, point_validity] = step(pointTracker, img); % implements klt, change to other implementation
keypoints = keypoints(:,point_validity);
newKeypoints = newKeypoints(point_validity,:)';

%% Estimate relative pose
keypointsHom = [keypoints; ones(1, size(keypoints,2))];
newKeypointsHom = [newKeypoints; ones(1, size(newKeypoints,2))];

E = estimateEssentialMatrix(keypointsHom, newKeypointsHom, K, K);
[R_10,u3] = decomposeEssentialMatrix(E);
[R_10, T_10] = disambiguateRelativePose(R_10, u3, keypointsHom, newKeypointsHom, K, K);

%% compute homogenous transforms
H_10 = [R_10, T_10;
       0, 0, 0, 1];          % frame 0 to 1
H_01 = H_10\eye(4);          % frame 1 to 0

H_W1 = H_W0*H_01;           % frame 1 to World

H_0W = H_W0\eye(4);         % frame World to 0
H_1W = H_W1\eye(4);         % frame World to 1

%% find new points to track
corners_i = detectHarrisFeatures(img, 'MinQuality', 0.2); %tryout, change with homemade or soln, SIFT??
strongestCorners_i = corners_i.selectStrongest(numNewFeaturesToFind);
strongestCornersCoord_i= strongestCorners_i.Location';

%% merge new points with existing points
newKeypoints = [newKeypoints, strongestCornersCoord_i];
[~,inds,~] = unique(round(newKeypoints)', 'rows', 'stable');
newKeypoints = newKeypoints(:,inds);

%hack to reinitialize??? :S
release(pointTracker);
initialize(pointTracker,newKeypoints',img);

%% return values
pose = H_W1;
state.tracker = pointTracker;
state.keypoints = newKeypoints;
state.landmarks = P_landmarks_W;

end