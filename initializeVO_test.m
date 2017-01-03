function [pose, state] = initializeVO_test(kitti_path,K)


%% Init
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 1500;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 4;

H_W1 = [0.999992295557242, -0.00258119695646088, -0.00295740569246971, -0.00390839692889836;
        0.00258386452115904, 0.999996258137531, 0.000898529394397742, -0.0120754601969495;
        0.00295507534492626, -0.000906164007372759, 0.999995223186840, 0.692742779058853;
        0, 0, 0, 1];

%% load data
ground_truth = load([kitti_path '/poses/00.txt']);
pose0 = [reshape(ground_truth(1,:)',4,3)';
        0, 0, 0, 1];

% kitti data from ex6
establishedKeypoints = load('./other_data/keypoints.txt')';
p_W_landmarks = load('./other_data/p_W_landmarks.txt')';

img0 = imread([kitti_path, '/00/image_0/', sprintf('%06d.png',0)]);

%% extract information
harris_score = harris(img0,harris_patch_size, harris_kappa);
potentialKeypoints = selectKeypoints(harris_score, num_keypoints, nonmaximum_supression_radius);
potentialKeypoints = setdiff(potentialKeypoints', establishedKeypoints', 'rows')';

establishedDescriptors = describeKeypoints(img0, establishedKeypoints, descriptor_radius);
potentialDescriptors = describeKeypoints(img0, potentialKeypoints, descriptor_radius);

establishedKeypoints = flipud(establishedKeypoints);
potentialKeypoints = flipud(potentialKeypoints);

potentialBearing = K\[potentialKeypoints; ones(1, size(potentialKeypoints,2))];
potentialBearing = pose0(1:3,1:3)*(potentialBearing./(ones(3,1)*sqrt(sum(potentialBearing.^2,1))));

H_01 = pose0\H_W1;

%% validation plot
figure(2),clf;
hold on;
plotPoseXY(gca,pose0);

scatter(p_W_landmarks(3,:), -p_W_landmarks(1,:),'.b');
axis equal;
drawnow;
%% return values
dummypose = pose0/H_01;
pose = pose0;

state.poses = [dummypose(:), pose0(:)];
state.landmarks = [p_W_landmarks;
                    ones(1,size(p_W_landmarks,2))];
                
state.establishedKeypoints = establishedKeypoints;
state.establishedDescriptors = establishedDescriptors;
state.establishedDescriptorsTransform = [zeros(4,size(establishedKeypoints,2));
                                        establishedKeypoints];

state.potentialKeypoints = potentialKeypoints;
state.potentialDescriptors = potentialDescriptors;
state.potentialBearingFirst = potentialBearing;
state.potentialKeypointsFirst = potentialKeypoints; % can be removed if not used for plotting
state.potentialPoseIndFirst = ones(1,size(potentialKeypoints,2));

end