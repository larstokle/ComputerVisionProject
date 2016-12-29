function [pose, state] = initializeVO(img0, img1, K0, K1, H_W0, numKeypoints)
%[pose, state] = initializeVO(img0, img1, K0, K1, H_W0):
%
%Finds keypoints in img0 and tracks them to img1.
%Then estimates the relative pose between img0 and img1.
%Makes all the tracked keypoints into landmarks.
%Updates the tracker with new keypoints
%   
%   Input: 
%       img0:   first grayscale image of sequence
%       img1:   first greyscale keyframe image of sequence
%       K0:     intrinsic matrix for camera that took img0
%       K1:     intrinsic matrix for camera that took img1
%       H_W0:   homogenous transformation matrix from frame 0 to world
%
%   Output:
%       pose:   pose at img1
%       state:  
    
%% Init
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 500;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 4;

%% find keypoints
harrisScore0 = harris(img0, harris_patch_size, harris_kappa);
harrisScore1 = harris(img1, harris_patch_size, harris_kappa);

corners0 = selectKeypoints(harrisScore0, num_keypoints, nonmaximum_supression_radius);
corners1 = selectKeypoints(harrisScore1, num_keypoints, nonmaximum_supression_radius);

descriptors0 = describeKeypoints(img0, corners0, descriptor_radius);
descriptors1 = describeKeypoints(img1, corners1, descriptor_radius);

% [descriptors0, corners0] = extractFeatures(img0, corners0);%, 'Method', 'Block', 'BlockSize', descriptorPatchSize);
% [descriptors1, corners1] = extractFeatures(img1, corners1);%, 'Method', 'Block', 'BlockSize', descriptorPatchSize);
% corners0 = corners0.Location';
% corners1 = corners1.Location';

corners0 = flipud(corners0);
corners1 = flipud(corners1);

%validation plot
figure(1);clf;
subplot(2,1,1);
title('keypoints in img0 tracked to img1');
imshow(img0);
hold on;
scatter(corners0(1,:),corners0(2,:),'rx');
%scatter(corners0(1,:), corners0(2,:),'rx','Linewidth',2);
%scatter(621,188/2,'g'); % plot coord testing

%% track keypoints from img0 in img1
matches = matchDescriptors(descriptors0, descriptors1, match_lambda);

% remove non matches
[~, ind0, ind1] = find(matches);
[~, ind0non, ind1non] = find(matches == 0);
corners0 = corners0(:,ind0);
descriptors0 = descriptors0(:,ind0);
potentialKeypoints = corners1(:,ind1non);
potentialDescriptors = descriptors1(:,ind1non);
corners1 = corners1(:,ind1);
descriptors1 = descriptors1(:,ind1);

%remove keypoints with to high distance
acceptedMatchInds = find(sum((corners0 - corners1).^2,1) < 300^2);
corners0 = corners0(:,acceptedMatchInds);
descriptors0 = descriptors0(:,acceptedMatchInds);
corners1 = corners1(:,acceptedMatchInds);
descriptors1 = descriptors1(:,acceptedMatchInds);

% corners0 = corners0.Location(indPairEstablished(:,1),:)';
% corners1 = corners1.Location(indPairEstablished(:,1),:)';


% validation plots
plot([corners0(1,:); corners1(1,:)],[corners0(2,:); corners1(2,:)],'g','Linewidth',1);
subplot(2,1,2);
title('tracked keypoints in img1');
imshow(img1);
hold on;
scatter(corners1(1,:), corners1(2,:),'rx','Linewidth',2);

%% get relative camera pose
%valid coords in 2D hom
corners0Hom = [corners0; ones(1, size(corners0,2))];
corners1Hom = [corners1; ones(1, size(corners0,2))];

%estimate hom trans
E = estimateEssentialMatrix(corners0Hom, corners1Hom, K0, K1);
[R_10,u3] = decomposeEssentialMatrix(E);
[R_10, T_10] = disambiguateRelativePose(R_10, u3, corners0Hom, corners1Hom, K0, K1);


%% homogenous transformation matrix
H_10 = [R_10, T_10;
       0, 0, 0, 1];          % frame 0 to 1
H_01 = H_10\eye(4);           % frame 1 to 0

H_W1 = H_W0*H_01;           % frame 1 to World

H_0W = H_W0\eye(4);
H_1W = H_W1\eye(4);

%validation plot
figure(2),clf;
hold on;
plotPoseXY(gca, H_W0);
plotPoseXY(gca, H_W1);
axis equal;

%% get landmarks in world coord
P_landmark_W = linearTriangulation(corners0Hom, corners1Hom, K0*H_0W(1:3,:), K1*H_1W(1:3,:));
P_landmark_1 = H_1W*P_landmark_W;
posZInds = find(P_landmark_1(3,:) > 0 & P_landmark_1(3,:) < 60);

corners1 = corners1(:,posZInds);
descriptors1 = descriptors1(:,posZInds);
P_landmark_W = P_landmark_W(:,posZInds);

%validation plot
scatter(P_landmark_W(1,:),P_landmark_W(2,:),'.b');
axis equal;


%% return values
pose = H_W1;

state.poses = H_W1(:);
state.establishedKeypoints = corners1;
state.establishedDescriptors = descriptors1;
state.landmarks = P_landmark_W;

state.potentialKeypoints = potentialKeypoints;
state.potentialDescriptors = potentialDescriptors;
state.potentialKeypointsFirst = potentialKeypoints;
state.potentialPoseIndFirst = ones(1,size(potentialKeypoints,2));
end
