function matches = matchDescriptorsEpiPolar(...
    descriptors1, descriptors2, keypoints1, keypoints2,...
    lambda, H_21, K, max_epipole_line_dist, max_dist)
%% matches = matchDescriptorsEpiPolar(...
%    descriptors1, descriptors2, keypoints1, keypoints2,...
%    lambda, H_12, K, max_epipole_line_dist, max_dist)
%
% Returns a 1xQ matrix where the i-th coefficient is the index of the
% descriptor1 which matches to the i-th descriptor2.
% The descriptor vectors are MxQ and MxD where M is the descriptor
% dimension and Q and D the amount of query and database descriptors
% respectively. matches(i) will be zero if there is no database descriptor
% with an SSD < lambda * min(SSD). No two non-zero elements of matches will
% be equal.

nonvalid = false(size(keypoints2,2),size(keypoints1,2));

if max_dist ~= 0
    keypointDist = pdist2(double(keypoints2)', double(keypoints1)','euclidean');
    nonvalid = nonvalid | (keypointDist > max_dist);
end

if max_epipole_line_dist ~= 0
    
    epipolar_distances = allVsAllEpipolarLineDistance(H_21,K,homogenize2D(keypoints1),homogenize2D(keypoints2));
    nonvalid = nonvalid | epipolar_distances > max_epipole_line_dist;
    
    %% === check this!  > NOTE: R_21 used to be R_12. Not checked if correct here.
    p2_far = projectPoints([R_21', -R_21'*T_21]*[p1_n * 120;ones(1, size(keypoints1, 2))],K);
    p2_close = projectPoints([R_21', -R_21'*T_21]*[p1_n * 1;ones(1, size(keypoints1, 2))],K);
    
    p2_min = min(cat(3,p2_far,p2_close),[],3);
    p2_max = max(cat(3,p2_far,p2_close),[],3);
    
    N1 = size(keypoints1,2);
    N2 = size(keypoints2,2);
    
    nonvalid = nonvalid | ~(ones(N2,1)*p2_min(1,:) < keypoints2(1,:)'*ones(1,N1) & keypoints2(1,:)'*ones(1,N1) < ones(N2,1)*p2_max(1,:))...
                        | ~(ones(N2,1)*p2_min(2,:) < keypoints2(2,:)'*ones(1,N1) & keypoints2(2,:)'*ones(1,N1) < ones(N2,1)*p2_max(2,:));
   %% ====== <
end

descriptorDist = pdist2(double(descriptors2)', double(descriptors1)', 'euclidean');
minDescDist = min(descriptorDist(:));
descriptorDist(nonvalid) = inf;

[minValidDescDist, matches] = min(descriptorDist,[],1);
matches(minValidDescDist > minDescDist*lambda) = 0;

unique_matches = zeros(size(matches));
[~,unique_match_idxs,~] = unique(matches, 'stable');
unique_matches(unique_match_idxs) = matches(unique_match_idxs);

matches = unique_matches;
end



%testing
function test()
n = 50
[~, ind] = datasample(1:N1,n,'replace',false);
figure(6);clf;
hold on;
for i = 1:n
plot([p2_close(1,ind(i)), p2_far(1,ind(i))],[p2_close(2,ind(i)), p2_far(2,ind(i))],'color',[(i-1)/(n-1),0,1-i/n]);
scatter(keypoints1(1,ind(i)),keypoints1(2,ind(i)),4,[(i-1)/(n-1),0,1-i/n]);
scatter(p2_close(1,ind(i)),p2_close(2,ind(i)),4,[0,0,0]);
scatter(p2_far(1,ind(i)),p2_far(2,ind(i)),4,[0,1,0]);
end
axis equal
p_epi = projectPoints([R_12', -R_12'*T_12]*[keypoints1_norm * 0;ones(1, size(keypoints1, 2))],K);
scatter(p_epi(1,:),p_epi(2,:),'cx')
end