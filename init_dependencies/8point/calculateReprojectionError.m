function [err,dir] = calculateReprojectionError(P,p,M)
    p_proj = M*P; % Project 3D points into image plane using projection matrix M
    p_proj = p_proj./p_proj(3,:); % Dehomogenize
    err_vec = p_proj - p;
    err = sqrt(sum((err_vec).^2,1)); % Error is euclidian distance
    dir = err_vec./err;
end