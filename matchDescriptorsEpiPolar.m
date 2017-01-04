function matches = matchDescriptorsEpiPolar(...
    descriptors1, descriptors2, keypoints1, keypoints2, lambda, H_12, K, max_epipole_line_dist, max_dist)
% Returns a 1xQ matrix where the i-th coefficient is the index of the
% database descriptor which matches to the i-th query descriptor.
% The descriptor vectors are MxQ and MxD where M is the descriptor
% dimension and Q and D the amount of query and database descriptors
% respectively. matches(i) will be zero if there is no database descriptor
% with an SSD < lambda * min(SSD). No two non-zero elements of matches will
% be equal.

if max_dist == 0
    max_dist = inf;
end

if max_epipole_line_dist == 0
    max_epipole_line_dist = inf;
end

T_12 = H_12(1:3,4);
R_12 = H_12(1:3,1:3);
normCoord2pix = K(1,1);

keypoints2_norm = K\[keypoints2; ones(1, size(keypoints2, 2))];
keypoints1_norm = K\[keypoints1; ones(1, size(keypoints1, 2))];

E = cross2matrix(T_12)*R_12;
epiPolarNormals = E'*keypoints1_norm;
epiPolarNormals = epiPolarNormals./(ones(3,1)*sqrt(sum(epiPolarNormals.^2,1)));

epiPolarDists = normCoord2pix*abs(keypoints2_norm'*epiPolarNormals);


keypointDist = pdist2(double(keypoints2)', double(keypoints1)','euclidean');
descriptorDist = pdist2(double(descriptors2)', double(descriptors1)', 'euclidean');
descriptorDist(keypointDist > max_dist | epiPolarDists > max_epipole_line_dist) = inf;

[minDescDist, matches] = min(descriptorDist,[],1);
matches(minDescDist > min(minDescDist)*lambda) = 0;

unique_matches = zeros(size(matches));
[~,unique_match_idxs,~] = unique(matches, 'stable');
unique_matches(unique_match_idxs) = matches(unique_match_idxs);

matches = unique_matches;

% [dists,matches] = pdist2(double(database_descriptors)', ...
%     double(query_descriptors)', 'euclidean', 'Smallest', 1);
% 
% sorted_dists = sort(dists);
% sorted_dists = sorted_dists(sorted_dists~=0);
% min_non_zero_dist = sorted_dists(1);
% 
% matches(dists >= lambda * min_non_zero_dist) = 0;
% 
% % remove double matches
% unique_matches = zeros(size(matches));
% [~,unique_match_idxs,~] = unique(matches, 'stable');
% unique_matches(unique_match_idxs) = matches(unique_match_idxs);
% 
% matches = unique_matches;

end

function M = cross2matrix(x)

M = [0,-x(3),x(2); x(3),0,-x(1);-x(2),x(1),0];

end

