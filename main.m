%% init parameters
VOpipe = 1; % 0: monocular p3p RANSAC, 1: monocular 1p-histogram with 8point essential matrix, 2: stereo VO, 3: Mono with stereo init

ds = 0; % 0: KITTI, 1: Malaga, 2: parking


makeVid = false;
videoFilename = 'VOplots';
fps = 4;

%% Setup
%plot setup
fh = figure(1);clf;drawnow;
set(fh, 'Color','white','Position', get(0,'Screensize'));
ax1 = axes('Position',[0,0,1,1]); set(ax1,'Box','off','Visible','off','Color','none');
ax2 = axes('Position',[0.075, 0.1, 0.4, 0.4]); hold(ax2,'on'); set(ax2, 'Box','on');
ax3 = axes('Position',[0.075, 0.6, 0.4, 0.3]); set(ax3, 'Box', 'off', 'Visible','off'); 
ax4 = axes('Position',[0.525, 0.6, 0.4, 0.3]); set(ax4, 'Box', 'off', 'Visible','off'); 
ax5 = axes('Position',[0.525, 0.1, 0.4, 0.4]); hold(ax5,'on'); set(ax5, 'Box','on', 'YaxisLocation','right');


plotAx.main = ax1;
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


% video setup
if makeVid
    vidObj = VideoWriter(videoFilename,'Motion JPEG AVI'); % Prepare video file
    vidObj.FrameRate = fps;
    open(vidObj);
    cleanupObj = onCleanup(@()endFunc(vidObj));
else
    cleanupObj = onCleanup(@()endFunc());
end

% path setup
%local_setup; %sets up the right folders
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
    frameStep = 1;
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
    frameStep = 1;
elseif ds == 2
    bootstrap_frames = [17 19];
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    frameStep = 1;
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
global state;
if VOpipe == 0 %0: monocular p3p RANSAC
    %% init
    [pose, state] = init(img0,img1,K);

    save('other_data/bootstrap_kitti_pose','pose');
    save('other_data/bootstrap_kitti_state','state');  
elseif VOpipe == 1
    [pose, state] = oneP_init(img1);
    
elseif VOpipe == 2 || VOpipe == 3
    assert(ds ~= 2,'Not stereo images for paring');
    state = [];
    for i = bootstrap_frames(1):frameStep:bootstrap_frames(2)
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
    if VOpipe ==3
        state = stereo2monoState(state,K);
    end
end

%% init continuous operation
range = (bootstrap_frames(2)+1):frameStep:last_frame;

% figure(2); clf; ax2 = gca; hold(ax2,'on');
% figure(5); clf; ax5 = gca; hold(ax5,'on');

th = text(ax1,0.5,0.95,sprintf('Frame %i',range(1)-1),'Editing','on','FontSize',16,'HorizontalAlignment','center');
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    set(th,'String',sprintf('Frame %i',i));
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
    if VOpipe == 0 || VOpipe == 3% 0: monocular p3p RANSAC
        
        [pose, state] = processFrame(image, K, pose, state,plotAx);
        

        plotPoseXY(ax2,pose);
        axis(ax2,'equal');
        plot(ax5,[state.poses(12+3,end-1), state.poses(12+3,end)],-[state.poses(12+2,end-1), state.poses(12+2,end)],'b')
        %pause;
        prev_img = image;
        
    elseif VOpipe == 1 % 1: monocular 1p-histogram with 8point essential matrix
        
        [pose, state] = processFrame3(image, K, pose, state,plotAx);
        %state = tracker(image,state);

        plotPoseXY(ax2,pose);
        axis(ax2,'equal');
        plot(ax5,[state.poses(12+3,end-1), state.poses(12+3,end)],-[state.poses(12+2,end-1), state.poses(12+2,end)],'b')
        

        prev_img = image;
        
    elseif VOpipe == 2 % 2: stereo VO
        
        [pose, state] = processStereo(image, image_r, K, baseline, state, plotAx);
         
    end
    toc
    refreshdata(ax1);
    drawnow;
    if makeVid
        writeVideo(vidObj, getframe(fh));
    end
    
end


% function endFunc(vidObj)
% if exist('vidObj','var')
%     close(vidObj);
% end 
% global state;
% save('state.mat','state');
% plotRotations(state);
% end