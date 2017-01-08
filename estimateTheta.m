function [theta_est, var_theta, inliers,H_C1_W] = estimateTheta(p1,p2,K,max_error)

assert(max_error > 0);

if size(p1,1) == 2
    p1 = homogenize2D(p1);
end
if size(p2,1) == 2
    p2 = homogenize2D(p2);
end

% Rotation from coordinate system in paper to our system
R = [0, 0, 1;-1, 0, 0;0, -1, 0];

p1_hom = p1;
p2_hom = p2;

p1 = K\p1;
p2 = K\p2;
p1 = R*p1;
p2 = R*p2;
p1 = normc(p1);
p2 = normc(p2);

theta = -2*atan((p2(2,:).*p1(3,:) - p1(2,:).*p2(3,:))./(p2(1,:).*p1(3,:) + p1(1,:).*p2(3,:)));
var_theta = var(theta);
theta_est = median(theta); % If above 10 degrees -> decreased inlier detection rate

if nargout >= 3
    E = [0, 0, sin(theta_est/2);
         0, 0, cos(theta_est/2);
         sin(theta_est/2), -cos(theta_est/2), 0];
    E = R'*E*R;

    F = inv(K')*E*inv(K);

    error = reprojectionErrorCostFn(F,p1_hom,p2_hom,K);
    inliers = error < max_error;
end

if nargout >= 4
    % Obtain extrinsic parameters (R,t) from E
    [Rots,u3] = decomposeEssentialMatrix(E);

    if nnz(inliers) > 0
        % Disambiguate among the four possible configurations
        [R_C1_W,T_C1_W] = disambiguateRelativePose(Rots,u3,p1_hom(:,inliers),p2_hom(:,inliers),K,K);
        H_C1_W = [R_C1_W , T_C1_W ; 0 0 0 1];
    else
        % You don't want to be here, but it can happen
        while nnz(inliers) == 0
            max_error = max_error*2;
            inliers =  error < max_error;
        end
        
        disp(['Warning: fitting model to ' num2str(nnz(inliers)) ' allowing reprojection error of ' num2str(max_error) ' as opposed to default ' num2str(max_error)]);
        [R_C1_W,T_C1_W] = disambiguateRelativePose(Rots,u3,p1_hom(:,inliers),p2_hom(:,inliers),K,K);
        H_C1_W = [R_C1_W , T_C1_W ; 0 0 0 1];
    end        
end

end
