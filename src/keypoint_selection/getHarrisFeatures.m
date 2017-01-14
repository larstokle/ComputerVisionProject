function [keypoints, descriptors] = getHarrisFeatures(image)
    % Harris parameters
    harris_patch_size = 9;
    harris_kappa = 0.08;
    num_keypoints = 2000;
    nonmaximum_supression_radius = 8;
    descriptor_radius = 9;

    scores = harris(image, harris_patch_size, harris_kappa);
    keypoints = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius);
    descriptors = describeKeypoints(image, keypoints, descriptor_radius);
    
    keypoints = flipud(keypoints);
end