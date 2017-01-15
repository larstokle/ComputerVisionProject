function plotRemovedXZrot(state)
H = reshape(state.poses,[4,4,size(state.poses,2)]);
N = size(state.poses,2);

H_new = zeros(size(H));
H_new(:,:,1) = eye(4);
for i = 2:N
    H_W1 = H(:,:,i-1);
    H_W2 = H(:,:,i);
    H_12 = H_W1^-1*H_W2;
    
    twist = homogMatrix2twist(H_12);
    twist([4,6]) = 0;
    
    H_12_new = twist2homogMatrix(twist);
    
    H_new(:,:,i) = H_new(:,:,i-1)*H_12_new;
end

figure(3);clf;
plot(squeeze(H_new(3,4,:)),squeeze(-H_new(1,4,:)));
title('Trajectory with camera X and Z rotations removed');
figure(4);clf;
plot(squeeze(H_new(3,4,:)),squeeze(-H_new(2,4,:)));
end