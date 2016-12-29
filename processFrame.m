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
match_lambda_est = 6;
match_lambda_pot = 4;

triangulationAngleThresh = 0.375*pi/180;
triangulationCosThresh = cos(triangulationAngleThresh);

poses = oldState.poses;
N = size(poses,2) + 1;

P_landmarks_W = oldState.landmarks;

establishedKeypoints = oldState.establishedKeypoints;
establishedDescriptors = oldState.establishedDescriptors;

potentialKeypoints = oldState.potentialKeypoints;
potentialDescriptors = oldState.potentialDescriptors;
potentialKeypointsFirst = oldState.potentialKeypointsFirst;
potentialPoseIndFirst = oldState.potentialPoseIndFirst;

%% find points in this frame to query
harrisScore = harris(img, harris_patch_size, harris_kappa);
newKeypoints = selectKeypoints(harrisScore, num_keypoints, nonmaximum_supression_radius);
newDescriptors = describeKeypoints(img, newKeypoints, descriptor_radius);
newKeypoints = flipud(newKeypoints);

%% track established points
matches = matchDescriptors(newDescriptors, establishedDescriptors, match_lambda_est);
[~, newInd, establishedInd] = find(matches);
trackedEstablishedKeypoints = newKeypoints(:,newInd);
trackedEstablishedDescriptors = newDescriptors(:,newInd);

trackedLandmarks = P_landmarks_W(:, establishedInd);

%remove established from new
[~, newIndNon, establishedIndNon] = find(matches == 0);
newKeypoints = newKeypoints(:,newIndNon);
newDescriptors = newDescriptors(:,newIndNon);

%validation plot established points
figure(3);clf;
imshow(img);
hold on;
scatter(establishedKeypoints(1,:), establishedKeypoints(2,:), 'yx','Linewidth',2');
plot([establishedKeypoints(1,establishedInd); trackedEstablishedKeypoints(1,:)],[establishedKeypoints(2,establishedInd); trackedEstablishedKeypoints(2,:)],'r','Linewidth',1);
title('tracked established keypoints from last frame');
hold on;
scatter(trackedEstablishedKeypoints(1,:), trackedEstablishedKeypoints(2,:),'rx','Linewidth',2);

%% Estimate relative pose

[H_CW, inliers] = ransacLocalization(trackedEstablishedKeypoints, trackedLandmarks, K);
fprintf('number of inliers found: %i\n',sum(inliers));


trackedEstablishedKeypoints = trackedEstablishedKeypoints(:,inliers);
trackedEstablishedDescriptors = trackedEstablishedDescriptors(:,inliers);
trackedLandmarks = trackedLandmarks(:,inliers);

%validate ransac
figure(3);
plot([establishedKeypoints(1,establishedInd(inliers)); trackedEstablishedKeypoints(1,:)],[establishedKeypoints(2,establishedInd(inliers)); trackedEstablishedKeypoints(2,:)],'g','Linewidth',1);

%% compute homogenous transforms
H_WC = H_CW\eye(4);         % frame World to 1

%% track potential points
matches = matchDescriptors(newDescriptors, potentialDescriptors, match_lambda_pot);
[~, newInd, potentialInd] = find(matches);
trackedPotentialKeypoints = newKeypoints(:,newInd);
trackedPotentialDescriptors = newDescriptors(:,newInd);

%potentialKeypointsFirst = potentialKeypointsFirst(:,potentialInd); %after
%plot for plotting all points first

%remove tracked potential points from new
[~, newIndNon, potentialIndNon] = find(matches == 0);
newKeypoints = newKeypoints(:,newIndNon);
newDescriptors = newDescriptors(:,newIndNon);


%validation plot potentials
figure(4);clf;
imshow(img);
hold on;
scatter(potentialKeypointsFirst(1,:), potentialKeypointsFirst(2,:), 'yx','Linewidth',2');
plot([potentialKeypointsFirst(1,potentialInd); trackedPotentialKeypoints(1,:)],[potentialKeypointsFirst(2,potentialInd); trackedPotentialKeypoints(2,:)],'g','Linewidth',1);
title('tracked potential keypoints from last frame');
hold on;
scatter(trackedPotentialKeypoints(1,:), trackedPotentialKeypoints(2,:),'rx','Linewidth',2);

potentialKeypointsFirst = potentialKeypointsFirst(:,potentialInd);
potentialPoseIndFirst = potentialPoseIndFirst(:,potentialInd);

%% triangulate new landmarks
trackedPotentialKeypointsHom = [trackedPotentialKeypoints; ones(1,size(trackedPotentialKeypoints,2))];
trackedPotentialKeypointsNorm = K\trackedPotentialKeypointsHom;
potentialKeypointsFirstHom = [potentialKeypointsFirst; ones(1, size(potentialKeypointsFirst,2))];
potentialKeypointsFirstNorm = K\potentialKeypointsFirstHom;

trackedLens = sqrt(sum(trackedPotentialKeypointsNorm.^2,1));
firstLens = sqrt(sum(potentialKeypointsFirstNorm.^2,1));

innerProd = sum(trackedPotentialKeypointsNorm.*potentialKeypointsFirstNorm,1)./(trackedLens.*firstLens);
goodPotentialInds = find(innerProd < triangulationCosThresh);

p1 = potentialKeypointsFirstHom(:,goodPotentialInds);
p1_poses_ind = potentialPoseIndFirst(goodPotentialInds);
p2 = trackedPotentialKeypointsHom(:,goodPotentialInds);

newLandmarks = zeros(4,size(p1,2));
for i= 1:size(p1,2)
    H_iW = reshape(poses(:,p1_poses_ind(i)),4,4)\eye(4);
    newLandmarks(:,i) = linearTriangulation(p1(:,i), p2(:,i), K*H_CW(1:3,:), K*H_iW(1:3,:));
end

newLandmarks_C = H_CW*newLandmarks;
posZInds = find(newLandmarks_C(3,:) > 0 & newLandmarks_C(3,:) < 60);
newLandmarks = newLandmarks(:,posZInds);
goodPotentialInds = goodPotentialInds(posZInds);

% ============ check this thing now!!! >
trackedLandmarks = [trackedLandmarks, newLandmarks];
trackedEstablishedKeypoints = [trackedEstablishedKeypoints, trackedPotentialKeypoints(:,goodPotentialInds)];
trackedEstablishedDescriptors = [trackedEstablishedDescriptors, trackedPotentialDescriptors(:,goodPotentialInds)];

stillPotentialInds = find(innerProd >= triangulationCosThresh);
trackedPotentialKeypoints = trackedPotentialKeypoints(:,stillPotentialInds);
trackedPotentialDescriptors = trackedPotentialDescriptors(:,stillPotentialInds);

potentialKeypointsFirst = potentialKeypointsFirst(:, stillPotentialInds);
potentialPoseIndFirst = potentialPoseIndFirst(stillPotentialInds);
% ============<
%validate plot new landmarks
figure(4);
scatter(p2(1,:),p2(2,:),'g','Linewidth',2);

figure(2);
scatter(newLandmarks(1,:), newLandmarks(2,:),'.r');
axis equal;

%% return values
pose = H_WC; 

state.poses = [poses, H_WC(:)];

state.establishedKeypoints = trackedEstablishedKeypoints;
state.establishedDescriptors = trackedEstablishedDescriptors;
state.landmarks = trackedLandmarks;

state.potentialKeypoints = [trackedPotentialKeypoints, newKeypoints];
state.potentialDescriptors = [trackedPotentialDescriptors, newDescriptors];
state.potentialKeypointsFirst = [potentialKeypointsFirst, newKeypoints];
state.potentialPoseIndFirst = [potentialPoseIndFirst, N*ones(1,size(newKeypoints,2))];


end