function b = calculateBearingVectors(keypoints,H_W_C,K)
    b = K\homogenize2D(keypoints);
    b = H_W_C(1:3,1:3)*(b./(ones(3,1)*sqrt(sum(b.^2,1))));
end