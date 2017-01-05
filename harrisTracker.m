% Track keypoints from previous frame to current frame
function [curr_keypoints, curr_descriptors, idx_of_tracked] = harrisTracker(prev_keypoints,prev_descriptors,image)

    % Harris parameters
    harris_patch_size = 9;
    harris_kappa = 0.08;
    num_keypoints = 2000;
    nonmaximum_supression_radius = 8;
    descriptor_radius = 9;
    match_lambda = 4;
    pixel_distance_limit = 100;
    debug = true;
    
    scores = harris(image, harris_patch_size, harris_kappa);
    curr_keypoints = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius);
    curr_descriptors = describeKeypoints(image, curr_keypoints, descriptor_radius);
    
    % NB: This is the opposite order of parameters from init. Init has (p0,p1) while it is (p1,p0) here 
    % Argument order: query_descriptors, database_descriptors, query_keypoints, database_keypoints, lambda, max_dist)
    matches = matchDescriptorsLocally(curr_descriptors,prev_descriptors,curr_keypoints,prev_keypoints,match_lambda,pixel_distance_limit);
    [~, idx_new, idx_of_tracked] = find(matches);
    
    if debug
        N_0 = size(prev_keypoints,2);
        N_1 = length(idx_new);
        dN = N_1 - N_0;
    
        disp(['Num matches: ' num2str(N_1) ' Num to track: ' num2str(N_0) ' Num lost track: ' num2str(-dN)]);
        figure(5);
        imshow(image); hold on;
        plotMatches(matches, curr_keypoints, prev_keypoints);
        hold off;
        pause(0.01);    
    end
    
    curr_keypoints = curr_keypoints(:,idx_new);
    curr_descriptors = curr_descriptors(:,idx_new);
end