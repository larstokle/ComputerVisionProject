function [theta_est, var_theta, inliers,H_C1_W] = estimateTheta(p1, p2,K,max_dist)
if size(p1,1) == 2
    p1 = homogenize2D(p1);
end
if size(p2,1) == 2
    p2 = homogenize2D(p2);
end

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
theta_est = median(theta);

disp('Theta mean and median');
disp(mean(theta)*180/pi);
disp(median(theta)*180/pi);

if nargout >= 3
    E = [0, 0, sin(theta_est/2);
         0, 0, cos(theta_est/2);
         sin(theta_est/2), -cos(theta_est/2), 0];
    E = R'*E*R;

    F = inv(K')*E*inv(K);

    dists = reprojectionErrorCostFn(F,p1_hom,p2_hom,K);
    inliers = dists < max_dist;
end

if nargout >= 4
    % Obtain extrinsic parameters (R,t) from E
    [Rots,u3] = decomposeEssentialMatrix(E);

    % Disambiguate among the four possible configurations
    [R_C1_W,T_C1_W] = disambiguateRelativePose(Rots,u3,p1_hom(:,inliers),p2_hom(:,inliers),K,K);
    H_C1_W = [R_C1_W , T_C1_W ; 0 0 0 1];
end

end
