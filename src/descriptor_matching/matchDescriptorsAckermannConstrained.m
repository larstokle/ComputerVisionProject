function [matches, theta, varTheta] = matchDescriptorsAckermannConstrained(...
    descriptors1, descriptors2, keypoints1, keypoints2,...
    lambda, K, max_epipole_line_dist, max_dist)
%% matches = matchDescriptorsAckermannConstrained(...
%    descriptors1, descriptors2, keypoints1, keypoints2,...
%    lambda, K, max_epipole_line_dist, max_dist)
%
% Returns a 1xQ matrix where the i-th coefficient is the index of the
% descriptor1 which matches to the i-th descriptor2.
% The descriptor vectors are MxQ and MxD where M is the descriptor
% dimension and Q and D the amount of query and database descriptors
% respectively. matches(i) will be zero if there is no database descriptor
% with an SSD < lambda * min(SSD). No two non-zero elements of matches will
% be equal.

debug = false;
if debug
    tic
    fprintf('\n === matchDescriptorsAckermannConstrained started === \n');
end
%init
nBins = 50;

% calculate descriptor distance
descriptorDist = pdist2(double(descriptors2)', double(descriptors1)', 'euclidean');

% remember best possible descriptor match in set
minDescDist = min(descriptorDist(:));

% find keypoints that are too far away each other in the image and remove them
if max_dist ~= 0
    keypointDist = pdist2(double(keypoints2)', double(keypoints1)','euclidean');
    descriptorDist(keypointDist > max_dist) = inf;
end

% find best valid match
matches = minUniqueMatchesBelow(descriptorDist, minDescDist*lambda);
if debug
    fprintf('#matches before pose estimation: %i\n',sum(matches > 0))
end

% find and remove keypoints that are too far from their epipolar lines
if max_epipole_line_dist ~= 0
    % find rotation transform
    [~, matchInd1, matchInd2] = find(matches);
    [theta, varTheta] = onePointHistogramVote(keypoints1(:,matchInd1), keypoints2(:,matchInd2), nBins, K);
    
    % rotation from normal image coordinates (z straight ahead),
    % to normal world coordinates (z straight up)
    R_xyz_zmxmy = [0, 0, 1;
        -1, 0, 0;
        0, -1, 0]; 
    
    % essential matrix in normal world coords
    E_21 = [0, 0, sin(theta/2);
         0, 0, cos(theta/2);
         sin(theta/2), -cos(theta/2), 0];
     
    % essential matrix in camera coords
    E_21 = R_xyz_zmxmy'*E_21*R_xyz_zmxmy;
    
    % fundamental matrix
    F_21 = inv(K')*E_21*inv(K);
    
    % find distances and remove those that are too far
    epipolar_distances = allVsAllEpipolarLineDistance(F_21,homogenize2D(keypoints1),homogenize2D(keypoints2));
    descriptorDist(epipolar_distances > max_epipole_line_dist) = inf;
    
    % improve match localization
    matches = minUniqueMatchesBelow(descriptorDist, minDescDist*lambda);
end
if debug
    fprintf('#matches before after estimation: %i\n',sum(matches > 0));
end
if debug
    toc
    fprintf(' === matchDescriptorsAckermannConstrained ended ===\n\n')
end
end
