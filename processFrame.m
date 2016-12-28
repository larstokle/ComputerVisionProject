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

%% init
numNewFeaturesToFind = 500;

P_landmarks_W = oldState.landmarks;
numLandmarks = size(P_landmarks_W,2);

establishedKeypoints = oldState.keypoints(:,1:numLandmarks);
potensialKeypoints = oldState.keypoints(:,(numLandmarks+1):end);

pointTracker = oldState.tracker;



%% track points
[trackedEstablishedKeypoints, point_validity] = step(pointTracker, img); % implements klt, change to other implementation
trackedEstablishedKeypoints = trackedEstablishedKeypoints(1:numLandmarks,:)';
validTrackedEstablishedKeypoints = trackedEstablishedKeypoints(:,point_validity(1:numLandmarks));

validEstablishedKeypoints = establishedKeypoints(:,point_validity(1:numLandmarks));
%keypoints = keypoints(:,point_validity);
%newKeypoints = trackedKeypoints(point_validity,:)';

%% Estimate relative pose
%validEstablishedKeypointsHom = [validEstablishedKeypoints; ones(1, size(validEstablishedKeypoints,2))];
%validTrackedEstablishedKeypointsHom = [validTrackedEstablishedKeypoints; ones(1, size(validTrackedEstablishedKeypoints,2))];
trackedLandmarks = P_landmarks_W(:, point_validity(1:numLandmarks));

[H_W1, inliers] = ransacLocalization(validTrackedEstablishedKeypoints, trackedLandmarks, K);
fprintf('number of inliers found: %i\n',sum(inliers));
% E = estimateEssentialMatrix(establishedKeypointsHom, trackedEstablishedKeypointsHom, K, K);
% [R_10,u3] = decomposeEssentialMatrix(E);
% [R_10, T_10] = disambiguateRelativePose(R_10, u3, establishedKeypointsHom, trackedEstablishedKeypointsHom, K, K);

%% compute homogenous transforms
H_1W = H_W1\eye(4);         % frame World to 1

%% find new points to track
corners_i = detectHarrisFeatures(img, 'MinQuality', 0.2); %tryout, change with homemade or soln, SIFT??
strongestCorners_i = corners_i.selectStrongest(numNewFeaturesToFind);
strongestCornersCoord_i= strongestCorners_i.Location';

%% merge new points with existing points
% newKeypoints = [newKeypoints, strongestCornersCoord_i];
% [~,inds,~] = unique(round(newKeypoints)', 'rows', 'stable');
% newKeypoints = newKeypoints(:,inds);
% 
% %hack to reinitialize??? :S
% release(pointTracker);
% initialize(pointTracker,newKeypoints',img);

%% return values
pose = H_W1;
state.tracker = pointTracker;
state.keypoints = trackedEstablishedKeypoints; %should be newKeypoints when done...
state.landmarks = P_landmarks_W;

end