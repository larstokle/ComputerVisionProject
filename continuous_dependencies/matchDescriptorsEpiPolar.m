function matches = matchDescriptorsEpiPolar(...
    query_descriptors, database_descriptors, query_keypoints, database_keypoints, lambda, relativePose, K, max_epipole_angle_error, max_dist)
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

if max_epipole_angle_error == 0
    max_epipole_angle_error = inf;
end

T = relativePose(1:3,4);
R = relativePose(1:3,1:3);

database_keypoints_norm = K\[database_keypoints; ones(1, size(database_keypoints, 2))];
query_keypoints_norm = K\[query_keypoints; ones(1, size(query_keypoints, 2))];

epiPolarNormals = R*cross2matrix(T)*database_keypoints_norm;
epiPolarNormals = epiPolarNormals./(ones(3,1)*sqrt(sum(epiPolarNormals,1)));

epiPolarDists = abs(epiPolarNormals'*(query_keypoints_norm./(ones(3,1)*sqrt(sum(query_keypoints_norm,1)))));


keypointDist = pdist2(double(database_keypoints)', double(query_keypoints)','euclidean');
descriptorDist = pdist2(double(database_descriptors)', double(query_descriptors)', 'euclidean');
descriptorDist(keypointDist > max_dist | epiPolarDists < sin(max_epipole_angle_error)) = inf;

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

