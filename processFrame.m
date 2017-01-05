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
match_lambda_est = 10;
pixel_motion_err_tol = 50;
reprojection_pix_tol = 10;

%tuning parameters potential landmarks
match_lambda_pot = 6;
max_epipole_line_dist = 10;
max_match_dist = 150;
triangulationAngleThresh = 2*pi/180;
triangulationCosThresh = cos(triangulationAngleThresh);



%extract variables from state
poses = oldState.poses;
N_frames = size(poses,2) + 1;

P_landmarks_W = oldState.landmarks;

establishedKeypoints = oldState.establishedKeypoints;
establishedDescriptors = oldState.establishedDescriptors;
establishedDescriptorsTransform = oldState.establishedDescriptorsTransform;
N_established = size(establishedKeypoints,2);
patch_radius = sqrt(size(establishedDescriptors,1));

potentialKeypoints = oldState.potentialKeypoints;
potentialDescriptors = oldState.potentialDescriptors;
potentialBearingFirst = oldState.potentialBearingFirst;
potentialKeypointsFirst = oldState.potentialKeypointsFirst;
potentialPoseIndFirst = oldState.potentialPoseIndFirst;
N_potential = size(potentialKeypoints, 2);

%% find points in this frame to query
harrisScore = harris(img, harris_patch_size, harris_kappa);
newKeypoints = selectKeypoints(harrisScore, num_keypoints, nonmaximum_supression_radius);
newDescriptors = describeKeypoints(img, newKeypoints, descriptor_radius);
newKeypoints = flipud(newKeypoints);

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
    establishedDescriptorsTransform(5:6,:) = projectPoints(P_landmarks_C_est(1:3,:), K);
else
    newEstablishedKeypointsEst = projectPoints(P_landmarks_C_est(1:3,:), K);
end

% tracking
if use_KLT
    % track with pyramid KLT
    establishedDescriptorsImg = reshape(establishedDescriptors,[patch_radius, patch_radius, N_established]);
	[ps, ds, valid_KLT] = KLTtracker(img, establishedDescriptorsTransform, establishedDescriptorsImg, KLT_match);
else
    matches = matchDescriptorsEpiPolar(establishedDescriptors, newDescriptors, newEstablishedKeypointsEst, newKeypoints, match_lambda_est,[],[],0,pixel_motion_err_tol);
end

%discard nonvalid tracks
if use_KLT
    trackedEstablishedKeypoints = ds(:,valid_KLT);
    trackedEstablishedDescriptors = establishedDescriptors(:,valid_KLT); %old dexcriptors used
    %trackedEstablishedDescriptors = describeKeypoints(img,flipud(round(trackedEstablishedKeypoints)), descriptor_radius); % if new descriptor is wanted
    trackedEstablishedDescriptorsTransform = ps(:,valid_KLT);
    
    trackedLandmarks = P_landmarks_W(:,valid_KLT);
    establishedKeypointsFound = establishedKeypoints(:,valid_KLT);
    
    [~, isNew] = setdiff(...
                round(newKeypoints'/nonmaximum_supression_radius),...
                round(trackedEstablishedKeypoints'/nonmaximum_supression_radius),...
                'rows');
    newKeypoints = newKeypoints(:,isNew);
    newDescriptors = newDescriptors(:,isNew);
    
else
    [~, establishedInd, newInd] = find(matches);
    
    trackedEstablishedKeypoints = newKeypoints(:,newInd);
    trackedEstablishedDescriptors = newDescriptors(:,newInd);
    
    establishedKeypointsFound = establishedKeypoints(:,establishedInd);
    trackedLandmarks = P_landmarks_W(:,establishedInd);
    
    [~, isNew] = setdiff(...
                round(newKeypoints'/nonmaximum_supression_radius),...
                round(trackedEstablishedKeypoints'/nonmaximum_supression_radius),...
                'rows');
    newKeypoints = newKeypoints(:,isNew);
    newDescriptors = newDescriptors(:,isNew);
end

%validation plot established points
%set(0,'CurrentFigure',3);clf;
cla(estAx);
imshow(img,'Parent',estAx);
hold(estAx,'on');
scatter(establishedKeypoints(1,:), establishedKeypoints(2,:), 'yx','Linewidth',2','Parent',estAx);
plot([establishedKeypointsFound(1,:); trackedEstablishedKeypoints(1,:)],[establishedKeypointsFound(2,:); trackedEstablishedKeypoints(2,:)],'r','Linewidth',1,'Parent',estAx);
title('tracked established keypoints from last frame','Parent',estAx);
scatter(trackedEstablishedKeypoints(1,:), trackedEstablishedKeypoints(2,:),'rx','Linewidth',2,'Parent',estAx);

establishedKeypoints = establishedKeypointsFound; %done here so that the old established can be plotted

%% Estimate relative pose

[H_CW, inliers] = ransacLocalization(trackedEstablishedKeypoints, trackedLandmarks, K, reprojection_pix_tol);
fprintf('RANSAC: #inliers: %i, #outliers: %i\n',sum(inliers),sum(inliers==0));

trackedEstablishedKeypoints = trackedEstablishedKeypoints(:,inliers);
trackedEstablishedDescriptors = trackedEstablishedDescriptors(:,inliers);
if use_KLT
    trackedEstablishedDescriptorsTransform = trackedEstablishedDescriptorsTransform(:,inliers);
end

trackedLandmarks = trackedLandmarks(:,inliers);
establishedKeypoints = establishedKeypoints(:,inliers);

%validate ransac
plot([establishedKeypoints(1,:); trackedEstablishedKeypoints(1,:)],[establishedKeypoints(2,:); trackedEstablishedKeypoints(2,:)],'g','Linewidth',1,'Parent',estAx);

%% compute homogenous transforms
H_WC = H_CW\eye(4);     % frame World to 1
H_last_C = H_W_last\H_WC;


%% track potential points
matches = matchDescriptorsEpiPolar(potentialDescriptors, newDescriptors, potentialKeypoints, newKeypoints, match_lambda_pot, H_last_C, K, max_epipole_line_dist, max_match_dist);

[~, potentialInd, newInd] = find(matches);
trackedPotentialKeypoints = newKeypoints(:,newInd);
trackedPotentialDescriptors = newDescriptors(:,newInd);
N_pot_tracked = size(trackedPotentialKeypoints,2);

%remove tracked potential points from new
[~, newIndNon] = setdiff(newKeypoints',trackedPotentialKeypoints','rows');
newKeypoints = newKeypoints(:,newIndNon);
newDescriptors = newDescriptors(:,newIndNon);

%calculate new bearings
newBearings = K\[newKeypoints; ones(1,size(newKeypoints,2))];
newBearings = H_WC(1:3,1:3)*(newBearings./(ones(3,1)*sqrt(sum(newBearings.^2,1))));

%validation plot potentials
%set(0,'CurrentFigure',4);clf;
cla(potAx);
imshow(img,'Parent',potAx);
hold(potAx,'on');
scatter(potentialKeypointsFirst(1,:), potentialKeypointsFirst(2,:), 'yx','Linewidth',2,'Parent',potAx);
plot([potentialKeypointsFirst(1,potentialInd); trackedPotentialKeypoints(1,:)],[potentialKeypointsFirst(2,potentialInd); trackedPotentialKeypoints(2,:)],'g','Linewidth',1, 'Parent',potAx);
scatter(trackedPotentialKeypoints(1,:), trackedPotentialKeypoints(2,:),'rx','Linewidth',2, 'Parent',potAx);

% remove nontracked points after ploting
potentialBearingFirst = potentialBearingFirst(:,potentialInd);
potentialKeypointsFirst = potentialKeypointsFirst(:,potentialInd);
potentialPoseIndFirst = potentialPoseIndFirst(:,potentialInd);

%% triangulate new landmarks

%calculate new bearing
trackedPotentialKeypointsHom = [trackedPotentialKeypoints; ones(1,size(trackedPotentialKeypoints,2))];
trackedPotentialKeypointsNorm = K\trackedPotentialKeypointsHom;
trackedLens = sqrt(sum(trackedPotentialKeypointsNorm.^2,1));
trackedPotentialKeypointsNormBearing = H_WC(1:3,1:3)*(trackedPotentialKeypointsNorm./(ones(3,1)*trackedLens));

% innerprod = cos(angle)... ==> low innerProd = large angle
innerProd = sum(trackedPotentialKeypointsNormBearing.*potentialBearingFirst,1);
goodPotentialInds = find(innerProd < triangulationCosThresh);

p1_bearings = potentialBearingFirst(:,goodPotentialInds);
p1_poses_ind = potentialPoseIndFirst(goodPotentialInds);
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
trackedLandmarks = [trackedLandmarks, newLandmarks];
trackedEstablishedKeypoints = [trackedEstablishedKeypoints, trackedPotentialKeypoints(:,goodPotentialInds)];
trackedEstablishedDescriptors = [trackedEstablishedDescriptors, trackedPotentialDescriptors(:,goodPotentialInds)];
if use_KLT
    trackedEstablishedDescriptorsTransform = [trackedEstablishedDescriptorsTransform,...
        [zeros(4,size(trackedPotentialKeypoints(:,goodPotentialInds),2));trackedPotentialKeypoints(:,goodPotentialInds)]];
end
    
stillPotentialInds = find(innerProd >= triangulationCosThresh);
trackedPotentialKeypoints = trackedPotentialKeypoints(:,stillPotentialInds);
trackedPotentialDescriptors = trackedPotentialDescriptors(:,stillPotentialInds);

potentialBearingFirst = potentialBearingFirst(:, stillPotentialInds);
potentialKeypointsFirst = potentialKeypointsFirst(:, stillPotentialInds);
potentialPoseIndFirst = potentialPoseIndFirst(stillPotentialInds);
% ============<

%validate plot new landmarks
%set(0,'CurrentFigure',4);
scatter(p2(1,:),p2(2,:),'g','Linewidth',2, 'Parent', potAx);

%set(0,'CurrentFigure',2);
scatter(newLandmarks(3,:), -newLandmarks(1,:),'.r', 'Parent', mapAx);
title(['tracked potential keypoints from last frame, #tracked = ',num2str(N_pot_tracked), ', #triangulated = ', num2str(size(newLandmarks,2))],'Parent',potAx);
axis(mapAx,'equal');

%% return values
pose = H_WC; 

state.poses = [poses, H_WC(:)];


state.establishedKeypoints = trackedEstablishedKeypoints;
state.establishedDescriptors = trackedEstablishedDescriptors;
if use_KLT
    state.establishedDescriptorsTransform = trackedEstablishedDescriptorsTransform;
else
    state.establishedDescriptorsTransform = [];
end
state.landmarks = trackedLandmarks;

state.potentialKeypoints = [trackedPotentialKeypoints, newKeypoints];
state.potentialDescriptors = [trackedPotentialDescriptors, newDescriptors];
state.potentialBearingFirst = [potentialBearingFirst, newBearings];
state.potentialKeypointsFirst = [potentialKeypointsFirst, newKeypoints];
state.potentialPoseIndFirst = [potentialPoseIndFirst, N_frames*ones(1,size(newKeypoints,2))];


end