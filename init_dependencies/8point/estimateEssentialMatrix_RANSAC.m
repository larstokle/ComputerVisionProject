function [E,inlier_mask] = estimateEssentialMatrix_RANSAC(p1, p2, K1, K2, num_iter, costFunction)
% estimateEssentialMatrix_normalized: estimates the essential matrix
% given matching point coordinates, and the camera calibration K
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%  - K1(3,3): calibration matrix of camera 1
%  - K2(3,3): calibration matrix of camera 2
%
% Output:
%  - E(3,3) : fundamental matrix
%
if nargin < 6
    costFunction = [];
end

[F,inlier_mask] = fundamentalEightPoint_RANSAC(p1, p2, num_iter, costFunction);

if nnz(inlier_mask>0)
    % Compute the essential matrix from the fundamental matrix given K
    E = K2'*F*K1;    
else
    E = [];
end

end
