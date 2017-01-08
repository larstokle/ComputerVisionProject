function [pose,state] = oneP_init(img)
    addpath('init_dependencies/harris/');

    % Pass keypoints and descriptors on
    [img_keypoints, img_descriptors] = getHarrisFeatures(img);
    
    pose = eye(4);

    state.poses = pose(:);
    
    % These must be present even though they are empty
    state.landmark_keypoints = [];
    state.landmark_descriptors = [];
    state.first_obs = [];

    state.candidate_keypoints = img_keypoints;
    state.candidate_descriptors = img_descriptors;

end