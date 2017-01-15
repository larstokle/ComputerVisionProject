function plotRemovedXZrot(state, plotOrig, ground_truth)
H = reshape(state.poses,[4,4,size(state.poses,2)]);
N = size(state.poses,2);

H_new = zeros(size(H));
H_new(:,:,1) = eye(4);
for i = 2:N
    H_W1 = H(:,:,i-1);
    H_W2 = H(:,:,i);
    H_12 = H_W1^-1*H_W2;
    
    twist = HomogMatrix2twist(H_12);
    twist([4,6]) = 0;
    
    H_12_new = twist2HomogMatrix(twist);
    
    H_new(:,:,i) = H_new(:,:,i-1)*H_12_new;
end

fh = figure(3);clf;
ax1 = subplot(1,2,1); hold on;
plot(squeeze(H_new(3,4,:)),squeeze(-H_new(1,4,:)),'DisplayName','X and Z rotations removed');
if exist('plotOrig','var') && plotOrig
    plot(squeeze(H(3,4,:)),squeeze(-H(1,4,:)),'DisplayName','VO estimate');
end
if exist('ground_truth','var')
    plot(ground_truth(3,1:N), -ground_truth(1,1:N),'DisplayName','Gound truth');
end
title('Trajectories in the plane');
%legend('-DynamicLegend');
xlabel('Camera Z-axis')
ylabel('Camera negative X-axis');


ax2 = subplot(1,2,2); hold on;
plot(squeeze(H_new(3,4,:)),squeeze(-H_new(2,4,:)),'DisplayName','X and Z rotations removed');
if exist('plotOrig','var') && plotOrig
    plot(squeeze(H(3,4,:)),squeeze(-H(2,4,:)),'DisplayName','VO estimate');
end
if exist('ground_truth','var')
    plot(ground_truth(3,1:N), -ground_truth(2,1:N),'DisplayName','Gound truth');
end
title('Height of trajectories in the plane');
lh2 = legend('-DynamicLegend');
set(lh2, 'Box', 'off');
xlabel('Camera Z-axis')
ylabel('Camera negative Y-axis');

axis(ax1,'equal');
axis(ax2, 'equal');
set(fh, 'Color', 'white', 'Position',get(0,'ScreenSize'));
fh.PaperPositionMode = 'auto';
fig_pos = fh.PaperPosition;
fh.PaperSize = [fig_pos(3) fig_pos(4)];
end