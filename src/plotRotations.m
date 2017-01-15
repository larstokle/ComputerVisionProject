function plotRotations(state)
figure(6);clf;
omegas = reshape(state.poses,[4,4,size(state.poses,2)]);
omegas = omegas(1:3,1:3,:);
for i = 1:size(state.poses,2)
omegas(:,:,i) = logm(omegas(:,:,i));
end
tvist = [];
for i = 1:size(state.poses,2)
temp = omegas(:,:,i);tvist(:,i) = temp([6,7,2]);
end
plot(tvist(1,:)*180/pi)
hold on
plot(tvist(2,:)*180/pi)
plot(tvist(3,:)*180/pi)
legend('X','Y','Z');
title('Rotation vector coords');
xlabel('Frame number');
ylabel('rotation coord [deg]');
end