function epipolar_dist = allVsAllEpipolarLineDistance(H_21,K,p1,p2)
%% epipolar_dist = allVsAllEpipolarLineDistance(...
%    H_21, K, p1, p2
%   
% Returns a matrix with epipolar distances between all points homogenous points p1,p2. The
% return value is a matrix of size len(p2)xlen(p1). H_21 is a homogenous
% transformation matrix from camera frame 1 to camera frame 1. K is the
% calibration matrix. 

    assert(size(p1,1)==3);
    assert(size(p2,1)==3);

    % Extract rotation and translation
    T_21 = H_21(1:3,4);
    R_21 = H_21(1:3,1:3);

    % Normalize
    p1_n = K\p1;
    p2_n = K\p2;
    
    % Calculate essential matrix
    E = cross2matrix(T_21)*R_21;
    
    epi_lines_1 = E'*p2_n;
    epi_lines_1 = epi_lines_1./(ones(3,1)*sqrt(sum(epi_lines_1(1:2,:).^2,1)));
    epi_dist_1 = (p1_n'*epi_lines_1).^2;
    
    epi_lines_2 = E*p1_n;
    epi_lines_2 = epi_lines_2./(ones(3,1)*sqrt(sum(epi_lines_2(1:2,:).^2,1)));
    epi_dist_2 = (p2_n'*epi_lines_2).^2;
    
    epipolar_dist = sqrt(epi_dist_1'+epi_dist_2);
    
    %% Test
    if false
        epipolar_dist_corr = zeros(size(p2,2),size(p1,2));

        for i = 1:size(p2_n,2)
            for j = 1:size(p1_n,2)
                epipolar_dist_corr(i,j) = distPoint2EpipolarLine(E,p1_n(:,j),p2_n(:,i));
            end
        end
        
        assert(all(all(abs(epipolar_dist-epipolar_dist_corr)<1e-6)));
    end    
end