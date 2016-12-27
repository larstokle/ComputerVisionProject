%% Setup
local_setup;
addpath('.\all_solns\04_8point', '.\all_solns\04_8point\triangulation', '.\all_solns\04_8point\8point');

ds = 0; % 0: KITTI, 1: Malaga, 2: parking
pose0 =[0, 0, 1, -0.2343818;
        -1, 0, 0, -0.141915;
        0, -1, 0, 4.291335;
        0, 0, 0, 1];

if ds == 0
    % need to set kitti_path to folder containing "00" and "poses"
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/00.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap
% need to set bootstrap_frames
bootstrap_frames_KITTI = [1;3]; %1 and 3 seems fine for now!
bootstrap_frames_Malaga = [1;3];
bootstrap_frames_parking = [1;3];

bootstrap_frames = bootstrap_frames_KITTI;

if ds == 0
    img0 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/00/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
else
    assert(false);
end

% kitti data from ex6
keypoints = load('./other_data/keypoints.txt')';
p_W_landmarks = load('./other_data/p_W_landmarks.txt')';

img0 = imread([kitti_path, '/00/image_0/', sprintf('%06d.png',0)]);
pointTracker = vision.PointTracker('BlockSize', [15,15]);
initialize(pointTracker, keypoints', img0);

state.tracker = pointTracker;
state.keypoints = keypoints;
state.landmarks = p_W_landmarks;

pose = pose0;

figure(1),clf;
hold on;
quiver(pose(1,4),pose(2,4),pose(1,1),pose(2,1),'Color','r'); %camera0 x-axis in w
quiver(pose(1,4),pose(2,4),pose(1,2),pose(2,2),'Color','g'); %camera0 y-axis in w
quiver(pose(1,4),pose(2,4),pose(1,3),pose(2,3),'Color','b'); %camera0 z-axis in w

scatter(p_W_landmarks(1,:),p_W_landmarks(2,:),'.b');
axis equal;

bootstrap_frames(2) = 0;
last_frame = 50;
%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    
    [pose, state] = processFrame(image, K, pose, state);
    
    quiver(pose(1,4),pose(2,4),pose(1,1),pose(2,1),'Color','r'); %camera0 x-axis in w
    quiver(pose(1,4),pose(2,4),pose(1,2),pose(2,2),'Color','g'); %camera0 y-axis in w
    quiver(pose(1,4),pose(2,4),pose(1,3),pose(2,3),'Color','b'); %camera0 z-axis in w  
   	drawnow;
    
    prev_img = image;
end
