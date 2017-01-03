function [F,inlier_mask] = fundamentalEightPoint_RANSAC(p1, p2, costFunction)
    
    if nargin <= 2 || isempty(costFunction)
        costFunction = @dist2epipolarLineCostFn;
    end

    % Parameters
    num_iterations = 10000; % Randomly selected
    rerun_on_inliers = true;
    debug = false;

    num_samples = 8; % We need eight points for the algorithm
    max_num_inliers = 0;
    idx = 1:length(p1);
    N = length(p1);
    best_guess = zeros(3,3);
    best_guess_inlier_mask = zeros(length(p1),1)';
        
    for i = 1:num_iterations
        % select 8 correspondences at random
        samples_idx = datasample(idx,num_samples,2,'Replace',false);
        
        % Fit model to these 8 correspondences
        F = fundamentalEightPoint_normalized(p1(:,samples_idx), p2(:,samples_idx));
        
        % Calculate error on all points and find inlier set
        [err,inlier_mask] = costFunction(F,p1,p2);
        num_inliers = nnz(inlier_mask);
        
        if num_inliers > max_num_inliers && num_inliers >= num_samples
            if rerun_on_inliers
                
                F_rerun = fundamentalEightPoint_normalized(p1(:,inlier_mask), p2(:,inlier_mask));
                [err_rerun,inlier_mask_rerun] = costFunction(F,p1,p2);
                
                if nnz(inlier_mask_rerun) > num_inliers
                   F = F_rerun;
                   err = err_rerun;
                   num_inliers = nnz(inlier_mask_rerun);
                end
            end
            
            max_num_inliers = num_inliers;
            best_guess = F;
            best_guess_inlier_mask = inlier_mask;
            
            if max_num_inliers == N
               break 
            end
        end
        
        if debug
            disp(['F8P: Iteration ' num2str(i) ' maxNumInliers: ' num2str(max_num_inliers) ' err: ' num2str(sum(err))]);
        end
        
    end
    
    F = best_guess;
    inlier_mask = best_guess_inlier_mask;
end

function [err, inlier_mask] = dist2epipolarLineCostFn(F,p1,p2)
        pixel_tolerance = 1; % Recomended in project description
        err = distPoint2EpipolarLine_pointwise(F,p1,p2);
        inlier_mask = err < pixel_tolerance;
end