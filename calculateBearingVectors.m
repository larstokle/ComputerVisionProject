function b = calculateBearingVectors(keypoints,H_W_C,K)
    assert(size(keypoints,1)==2)
    b = K\[keypoints; ones(1,size(keypoints,2))];
    b = H_W_C(1:3,1:3)*(b./(ones(3,1)*sqrt(sum(b.^2,1))));
end