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
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 500;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 4;

P_landmarks_W = oldState.landmarks;

establishedKeypoints = oldState.establishedKeypoints;
establishedDescriptors = oldState.establishedDescriptors;

% potensialKeypoints = oldState.potensialKeypoints;
% potensialDescriptors = oldState.potensialDescriptors;

%% track points
harrisScore = harris(img, harris_patch_size, harris_kappa);
newKeypoints = selectKeypoints(harrisScore, num_keypoints, nonmaximum_supression_radius);
newDescriptors = describeKeypoints(img, newKeypoints, descriptor_radius);
matches = matchDescriptors(establishedDescriptors, newDescriptors, match_lambda);
[~, establishedInd, newInd] = find(matches);
trackedEstablishedKeypoints = flipud(newKeypoints(:,newInd));
trackedEstablishedDescriptors = newDescriptors(:,newInd);

trackedLandmarks = P_landmarks_W(:, establishedInd);

%validation plot
figure(3);clf;
imshow(img);
hold on;
scatter(establishedKeypoints(1,:), establishedKeypoints(2,:), 'yx','Linewidth',2');
plot([establishedKeypoints(1,establishedInd); trackedEstablishedKeypoints(1,:)],[establishedKeypoints(2,establishedInd); trackedEstablishedKeypoints(2,:)],'g','Linewidth',1);
title('tracked keypoints from last frame');
hold on;
scatter(trackedEstablishedKeypoints(1,:), trackedEstablishedKeypoints(2,:),'rx','Linewidth',2);


%% Estimate relative pose

[H_CW, inliers] = ransacLocalization(trackedEstablishedKeypoints, trackedLandmarks, K);
fprintf('number of inliers found: %i\n',sum(inliers));


trackedEstablishedKeypoints = trackedEstablishedKeypoints(:,inliers);
trackedEstablishedDescriptors = trackedEstablishedDescriptors(:,inliers);
trackedLandmarks = trackedLandmarks(:,inliers);

%M_C_W = estimatePoseDLT(trackedEstablishedKeypoints', trackedLandmarks(1:3,:)', K);


%% compute homogenous transforms
H_WC = H_CW\eye(4);         % frame World to 1

%% find new points to track


%% merge new points with existing points

%% return values
pose = H_WC; 

state.establishedKeypoints = trackedEstablishedKeypoints;
state.establishedDescriptors = trackedEstablishedDescriptors;
state.landmarks = trackedLandmarks;

%potensialKeypoints = potensialKeypoints;
%potensialDescriptors = potensialDescriptors;

end