%% Setup
figure(2); clf; ax2 = gca; hold(ax2,'on');
figure(3); clf; ax3 = gca;
figure(4); clf; ax4 = gca;
figure(5); clf; ax5 = gca; hold(ax5,'on');

local_setup;
addpath('continuous_dependencies/');
use_saved_bootstrap = false;

ds = 0; % 0: KITTI, 1: Malaga, 2: parking

if ds == 0
    bootstrap_frames = [85 150];
    baseline = 0.54;
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
if ds == 0
    if use_saved_bootstrap
        load('other_data/bootstrap_kitti_pose');
        load('other_data/bootstrap_kitti_state');
    else
        
        for i = bootstrap_frames(1):bootstrap_frames(2)
            bootstrap_img_l(:,:,i-bootstrap_frames(1)+1) = imread([kitti_path '/00/image_0/' ...
                sprintf('%06d.png',i)]);
            bootstrap_img_r(:,:,i-bootstrap_frames(1)+1) = imread([kitti_path '/00/image_1/' ...
                sprintf('%06d.png',i)]);
        end
        
        img0 = bootstrap_img_l(:,:,1);
        img1 = bootstrap_img_r(:,:,end);
%         img0 = imread([kitti_path '/00/image_0/' ...
%             sprintf('%06d.png',bootstrap_frames(1))]);
%         img1 = imread([kitti_path '/00/image_0/' ...
%             sprintf('%06d.png',bootstrap_frames(2))]);
        
        %% init
        disp('Start kitti init');
        [pose, state] = init(img0,img1,K);
        disp('Kitti init finished');

        save('other_data/bootstrap_kitti_pose','pose');
        save('other_data/bootstrap_kitti_state','state');        
     end
    
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
    
    tic;
    [pose, state] = processFrame(image, K, pose, state);
    toc;
    
    plotPoseXY(ax2,pose);
    plot(ax5,[state.poses(12+3,end-1), state.poses(12+3,end)],-[state.poses(12+2,end-1), state.poses(12+2,end)],'b')
   	drawnow;
    %pause;
    prev_img = image;
end

%% aftermath

% plotting of the rotation
figure(6);clf;
omegas = reshape(state.poses,[4,4,size(state.poses,2)]);
omegas = omegas(1:3,1:3,:);
for i = 1:size(state.poses,2)
omgegas(:,:,i) = logm(omegas(:,:,i));
end
tvist = [];
for i = 1:size(state.poses,2)
temp = omegas(:,:,i);tvist(:,i) = temp([6,7,2]);
end
plot(tvist(1,:))
hold on
plot(tvist(2,:))
plot(tvist(3,:))
legend('X','Y','Z');
title('rotation vector coords');
