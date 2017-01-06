%% Setup environment

local_setup;
addpath('continuous_dependencies/');

bootstrap_frames = [1 3];

assert(exist('kitti_path', 'var') ~= 0);
ground_truth = load([kitti_path '/poses/00.txt']);
pose0 = [reshape(ground_truth(1,:)',4,3)';
        0, 0, 0, 1];
ground_truth = ground_truth(:, [end-8 end]);
last_frame = 4540;
K = [7.188560000000e+02 0 6.071928000000e+02
    0 7.188560000000e+02 1.852157000000e+02
    0 0 1];

img0 = imread([kitti_path '/00/image_0/' ...
    sprintf('%06d.png',bootstrap_frames(1))]);
img1 = imread([kitti_path '/00/image_0/' ...
    sprintf('%06d.png',bootstrap_frames(2))]);

load('other_data/bootstrap_kitti_pose');

H = pose; % Estimated from init

%% Extract harris features
% Harris parameters
harris_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 2000;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 7;

harrisScore0 = harris(img0, harris_patch_size, harris_kappa);
harrisScore1 = harris(img1, harris_patch_size, harris_kappa);

corners0 = selectKeypoints(harrisScore0, num_keypoints, nonmaximum_supression_radius);
corners1 = selectKeypoints(harrisScore1, num_keypoints, nonmaximum_supression_radius);

descriptors0 = describeKeypoints(img0, corners0, descriptor_radius);
descriptors1 = describeKeypoints(img1, corners1, descriptor_radius);

% (row,col) -> (x,y)
corners0 = flipud(corners0);
corners1 = flipud(corners1);

match_lambda_pot = 7;
max_epipole_line_dist = 15;
max_match_dist = 150;

H_21 = H^-1; % H is from 2 to world. We want the opposite.

matchDescriptorsEpiPolar(descriptors0(:,1:3), descriptors1(:,1:4), corners0(:,1:3), corners1(:,1:4), match_lambda_pot, H_21, K, max_epipole_line_dist, max_match_dist);