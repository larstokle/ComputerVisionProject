%% init parameters
VOpipe = 0; % 0: monocular p3p RANSAC, 1: monocular 1p-histogram with 8point essential matrix, 2: stereo VO

ds = 0; % 0: KITTI, 1: Malaga, 2: parking

%% Setup
fh = figure(1);clf;
set(fh, 'Color','white','Position', get(0,'Screensize'));
ax2 = axes('Position',[0.075, 0.1, 0.4, 0.4]); hold(ax2,'on'); set(ax2, 'Box','on');
ax3 = axes('Position',[0.075, 0.6, 0.4, 0.3]); set(ax3, 'Box', 'off', 'Visible','off'); 
ax4 = axes('Position',[0.525, 0.6, 0.4, 0.3]); set(ax4, 'Box', 'off', 'Visible','off'); 
ax5 = axes('Position',[0.525, 0.1, 0.4, 0.4]); hold(ax5,'on'); set(ax5, 'Box','on', 'YaxisLocation','right');

plotAx.map = ax2;
plotAx.lndmrk = ax3;
plotAx.cndt = ax4;
plotAx.height = ax5;

title(ax2, 'Landmarks and poses');
xlabel(ax2, 'Camera Z-axis');
ylabel(ax2, 'Camera negative X-axis');

title(ax5, 'Height');
xlabel(ax5, 'Camera Z-axis');
ylabel(ax5, 'Camera negative Y-axis');

% figure(2); clf; ax2 = gca; hold(ax2,'on');
% figure(3); clf; ax3 = gca;
% figure(4); clf; ax4 = gca;
% figure(5); clf; ax5 = gca; hold(ax5,'on');

local_setup; %sets up the right folders
addpath(genpath([pwd,'\src']));
addpath('inits');
addpath('process_frames');

if ds == 0
    bootstrap_frames = [1 3];
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
    bootstrap_frames = [1 3];
    baseline = 0.119471;
    
    % Path containing the many files of Malaga 7.
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    right_images = images(4:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    bootstrap_frames = [1 3];
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% init bootstrap 
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

%% run bootstrap
if VOpipe == 0 %0: monocular p3p RANSAC
    %% init
    disp('Start kitti init');
    [pose, state] = init(img0,img1,K);
    disp('Kitti init finished');

    save('other_data/bootstrap_kitti_pose','pose');
    save('other_data/bootstrap_kitti_state','state');  
elseif VOpipe == 1
    [pose, state] = oneP_init(img1);
    
elseif VOpipe == 2
    assert(ds ~= 2,'Not stereo images for paring');
    state = [];
    for i = bootstrap_frames(1):bootstrap_frames(2)
        if ds == 0
            img_l = imread([kitti_path '/00/image_0/' ...
                sprintf('%06d.png',i)]);
            img_r = imread([kitti_path '/00/image_1/' ...
                sprintf('%06d.png',i)]);
        elseif ds == 1
            img_l = rgb2gray(imread([malaga_path ...
                '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
                left_images(i).name]));
            img_r = rgb2gray(imread([malaga_path ...
                '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
                right_images(i).name]));
        end
        [pose,state] = processStereo(img_l, img_r, K, baseline, state, plotAx);
    end
end

%% init continuous operation
range = (bootstrap_frames(2)+1):last_frame;

% figure(2); clf; ax2 = gca; hold(ax2,'on');
% figure(5); clf; ax5 = gca; hold(ax5,'on');

for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/00/image_0/' sprintf('%06d.png',i)]);
        if VOpipe==2
           image_r = imread([kitti_path '/00/image_1/' sprintf('%06d.png',i)]);
        end
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
        image_r = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            right_images(i).name]));
         
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    
    tic;
%% run continous operation
    if VOpipe == 0 % 0: monocular p3p RANSAC
        
        [pose, state] = processFrame(image, K, pose, state,plotAx);
        

        plotPoseXY(ax2,pose);
        axis(ax2,'equal');
        plot(ax5,[state.poses(12+3,end-1), state.poses(12+3,end)],-[state.poses(12+2,end-1), state.poses(12+2,end)],'b')
        drawnow;
        %pause;
        prev_img = image;
        
    elseif VOpipe == 1 % 1: monocular 1p-histogram with 8point essential matrix
        
        [pose, state] = processFrame3(image, K, pose, state);
        %state = tracker(image,state);

        plotPoseXY(ax2,pose);
        axis(ax2,'equal');
        plot(ax5,[state.poses(12+3,end-1), state.poses(12+3,end)],-[state.poses(12+2,end-1), state.poses(12+2,end)],'b')
        drawnow;

        prev_img = image;
        
    elseif VOpipe == 2 % 2: stereo VO
        
        [pose, state] = processStereo(image, image_r, K, baseline, state, plotAx);
         
    end
    toc
end

%% aftermath
