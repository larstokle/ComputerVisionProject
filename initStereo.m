function [T, S] = initStereo(img_l,img_r,K, baseline)
    assert(sum(size(img_l) == size(img_r)) ~= 0);
    
    img_size = size(img_l);
    N_imgs = img_size(3);
    img_size = img_size(1:2);
    
    harris_patch_size = 9;
    harris_kappa = 0.08;
    num_keypoints = 1000;
    nonmaximum_supression_radius = 8;
    descriptor_radius = 9;
    r = descriptor_radius;
    match_lambda = 4;
    max_corner_dist = 200; 
    max_epipolar_dist = 25;
    
    min_disp = 2;
    max_max_disp = 60;
    patch_size = (2*r+1);
    
    reprojection_pix_tol = 10;
    
    
    harrisScore = zeros(size(img_l));
    corners = zeros(2,num_keypoints,N_imgs);
    descriptors = zeros((2*descriptor_radius + 1)^2, num_keypoints, N_imgs);
    landmarks = zeros(3,num_keypoints,N_imgs);
    
    Hs = zeros(4,4,N_imgs);
    Hs(:,:,1) = eye(4);
    
    for i = 1:size(img_l,3)
        harrisScore(:,:,i) = harris(img_l(:,:,i), harris_patch_size, harris_kappa);
        corners(:,:,i) = selectKeypoints(harrisScore(:,:,i), num_keypoints, nonmaximum_supression_radius);
        descriptors(:,:,i) = describeKeypoints(img_l(:,:,i), corners(:,:,i), descriptor_radius);
        corners(:,:,i) = flipud(corners(:,:,i));
        for j = 1:num_keypoints
            row = corners(2,j,i);
            col = corners(1,j,i);
            max_disp = max_max_disp + min([col-r-max_max_disp-1,0]);
            if max_disp <= min_disp*2
                
                continue;
            end
            
            padedImg = padarray(img_r(:,:,i),[r 0]);
            padedImg = padarray(padedImg,[0, r], 'post');
            right_strip = single(padedImg(...
                ((row-r):(row+r)) + r , (col-r-max_disp):(col+r-min_disp)));
            
            lpvec = single(descriptors(:,j,i));
            rsvecs = single(zeros(patch_size^2, max_disp - min_disp + 1));
            for k = 1:patch_size
                rsvecs(((k-1)*patch_size+1):(k*patch_size), :) = ...
                    right_strip(:, k:(max_disp - min_disp + k));
            end
            
            ssds = pdist2(lpvec', rsvecs', 'squaredeuclidean');
            [min_ssd, neg_disp] = min(ssds);

            
            if (nnz(ssds <= 1.5 * min_ssd) < 3 & neg_disp ~= 1 & ...
                neg_disp ~= length(ssds))

                x = [neg_disp-1 neg_disp neg_disp+1];
                p = polyfit(x, ssds(x), 2);
                disparity = max_disp + p(2)/(2*p(1));
            else
                disparity = max_disp - neg_disp;
            end
%             col_r = col - disparity;
            landmarks(:,j,i) = disparityToPoint(disparity,K,baseline,corners(:,j,i));
%             corners_z_values(j,i) = [1,0]*((K\[col, -col_r; row, -row; 1, -1])\[0; 0; baseline]);
        end
        
        if i > 1
            matches =  matchDescriptorsAckermannConstrained(descriptors(:,:,i-1), descriptors(:,:,i),...
                                                            corners(:,:,i-1), corners(:,:,i),...
                                                            match_lambda, K, max_epipolar_dist, max_corner_dist);
                                                        
%             matches = matchDescriptors(descriptors(:,:,i-1), descriptors(:,:,i), match_lambda);
            [~, ind0, ind1] = find(matches);
%             p_pix_hom = [corners(:,ind0,i-1); ones(1,size(ind0,2))];
%             p_CM1 = K\p_pix_hom.*(ones(3,1)*corners_z_values(ind0,i-1)');
%             tracked_landmarks = Hs(:,:,i-1)\[p_CM1; ones(1,size(ind0,2))];
            tracked_landmarks = landmarks(:,ind0,i-1);
            ok_lndmrks = [0,0,1,0]*(Hs(:,:,i-1)\[tracked_landmarks; ones(1,size(ind0,2))]) > 1;
            [H_CW, inliers] = ransacLocalization(corners(:,ind1(ok_lndmrks),i), tracked_landmarks(:,ok_lndmrks), K, reprojection_pix_tol);
            Hs(:,:,i) = H_CW\eye(4);
            figure(3);clf;
            imshow(img_l(:,:,i));
            hold on;
            plotMatches(matches,flipud(corners(:,:,i-1)),flipud(corners(:,:,i)));
            scatter(corners(1,ind1,i),corners(2,ind1,i),'r')
            pause;
        end
        landmarks(:,:,i) = eye(3,4)*Hs(:,:,i)*[landmarks(:,:,i); ones(1,num_keypoints)];
        figure(2); hold on
        plotPoseXY(gca,Hs(:,:,i));
        scatter(landmarks(3,:,i), -landmarks(1,:,i),'.b', 'Parent', gca);
        drawnow;
    end   
end