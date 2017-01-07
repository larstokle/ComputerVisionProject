function epipolar_dist = allVsAllEpipolarLineDistance(F_21,p1,p2)
%% epipolar_dist = allVsAllEpipolarLineDistance(F_21, p1, p2)
%   
% Returns a matrix with epipolar distances between all homogenous points p1,p2. The
% returned value is a matrix of size len(p2) x len(p1).
% F_21 is the fundamental matrix matrix from camera frame 1 to camera frame 2.
%   E_21 = t_21 cross R_21
%   F_21 = inv(K')*E_21*inv(K)

    assert(size(p1,1)==3);
    assert(size(p2,1)==3);
    
    epi_lines_1 = F_21'*p2; % epipolar normal 1
    epi_lines_1 = epi_lines_1./(ones(3,1)*sqrt(sum(epi_lines_1(1:2,:).^2,1))); %normalize
    epi_dist_1 = (p1'*epi_lines_1).^2;
    
    epi_lines_2 = F_21*p1; % epipolar normal 2
    epi_lines_2 = epi_lines_2./(ones(3,1)*sqrt(sum(epi_lines_2(1:2,:).^2,1))); %normalize
    epi_dist_2 = (p2'*epi_lines_2).^2;
    
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