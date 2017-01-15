function plotRelativeRotations(state)

H = reshape(state.poses,[4,4,size(state.poses,2)]);
N = size(state.poses,2);

twist = zeros(3,N);

for i = 2:N
    H_W1 = H(:,:,i-1);
    H_W2 = H(:,:,i);
    H_12 = H_W1^-1*H_W2;
    
    R = H_12(1:3,1:3);
    omega = logm(R); % Skew matrix
    
    twist(:,i) = omega([6,7,2]);
end    
    
figure(6);clf; 
plot(twist(1,:)*180/pi)
hold on
plot(twist(2,:)*180/pi)
plot(twist(3,:)*180/pi) 
legend('X','Y','Z');
title('Rotation vector coords. H_{i-1,i} for each frame i');
xlabel('Frame number');
ylabel('rotation coord [deg]');
ylim([-4 4]);

end