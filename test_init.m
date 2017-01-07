local_setup;
addpath('continuous_dependencies/');

bootstrap_frames = [1 3];

% need to set kitti_path to folder containing "00" and "poses"
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

disp('Start kitti init');
[pose, state] = init(img0,img1,K);
disp('Kitti init finished');