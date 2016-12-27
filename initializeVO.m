function [pose, state] = initializeVO(img0, img1, K0, K1, H_W0)
%initializeVO: finds position of second image and makes a list of landmarks
%   
%       Input: 
%           img0:   first grayscale image of sequence
%           img1:   first greyscale keyframe image of sequence
%           K0:     intrinsic matrix for camera that took img0
%           K1:     intrinsic matrix for camera that took img1
%           H_W0:   homogenous transformation matrix from frame 0 to world
%
%       Output:
%           pose:   pose at img1
%           state:  
    
%% Init
k = 500;

%% find keypoints in img0
corners = detectHarrisFeatures(img0, 'MinQuality', 0.2); %tryout, change with homemade or soln, SIFT??
strongestCorners = corners.selectStrongest(k);
strongestCornersCoord = strongestCorners.Location;

%validation plot
figure(1);clf;
subplot(2,1,1);
title('keypoints in img0 tracked to img1');
imshow(img0);
hold on;
scatter(strongestCornersCoord(:,1), strongestCornersCoord(:,2),'rx','Linewidth',2);
%scatter(621,188/2,'g'); % plot coord testing

%% track keypoints from img0 in img1
pointTracker = vision.PointTracker('BlockSize', [15,15]);
initialize(pointTracker, strongestCornersCoord, img0);
[trackedPoints, point_validity] = step(pointTracker, img1); % implements klt, change 

%valid coords
validStrongestCornersCoord = strongestCorners.Location(point_validity,:)';
validTrackedPoints = trackedPoints(point_validity,:)';

% validation plots
plot([validStrongestCornersCoord(1,:); validTrackedPoints(1,:)],[validStrongestCornersCoord(2,:); validTrackedPoints(2,:)],'g','Linewidth',1);
subplot(2,1,2);
title('tracked keypoints in img1');
imshow(img1);
hold on;
scatter(validTrackedPoints(1,:),validTrackedPoints(2,:),'rx','Linewidth',2);

%% get relative camera pose
%valid coords in 2D hom
validStrongestCornersCoordHom = [validStrongestCornersCoord; ones(1, sum(point_validity))];
%validStrongestCornersCoordHom = validStrongestCornersCoordHom./repmat(validStrongestCornersCoordHom(3,:),3,1);
validTrackedPointsHom = [validTrackedPoints; ones(1, sum(point_validity))];
%validTrackedPointsHom = validTrackedPointsHom./repmat(validTrackedPointsHom(3,:),3,1);

%estimate hom trans
E = estimateEssentialMatrix(validStrongestCornersCoordHom, validTrackedPointsHom, K0, K1);
[R_10,u3] = decomposeEssentialMatrix(E);
[R_10, T_10] = disambiguateRelativePose(R_10, u3, validStrongestCornersCoordHom, validTrackedPointsHom, K0, K1);


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
quiver(H_W0(1,4),H_W0(2,4),H_W0(1,1),H_W0(2,1),'Color','r'); %camera0 x-axis in w
quiver(H_W0(1,4),H_W0(2,4),H_W0(1,2),H_W0(2,2),'Color','g'); %camera0 y-axis in w
quiver(H_W0(1,4),H_W0(2,4),H_W0(1,3),H_W0(2,3),'Color','b'); %camera0 z-axis in w

quiver(H_W1(1,4),H_W1(2,4),H_W1(1,1),H_W1(2,1),'Color','r'); %camera1 x-axis in w
quiver(H_W1(1,4),H_W1(2,4),H_W1(1,2),H_W1(2,2),'Color','g'); %camera1 y-axis in w
quiver(H_W1(1,4),H_W1(2,4),H_W1(1,3),H_W1(2,3),'Color','b'); %camera1 z-axis in w
axis equal;

%% get landmarks in world coord
P_landmark_W = linearTriangulation(validStrongestCornersCoordHom, validTrackedPointsHom, K0*H_0W(1:3,:), K1*H_1W(1:3,:));

%validation plot
scatter(P_landmark_W(1,:),P_landmark_W(2,:),'.b');
axis equal;

%% update tracker
%hack to reinitialize??? :S
release(pointTracker);
initialize(pointTracker,validTrackedPoints',img1);

%% return values
pose = H_W1;

%maybe something else in state
state.landmarks = P_landmark_W;
state.keypoints = validTrackedPoints;
state.tracker = pointTracker;
%state.lastCorrespondenceLandmarkIndex = (1:size(validTrackedPoints,2))';
end
