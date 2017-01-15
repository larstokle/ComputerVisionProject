imgNrVec = [1376];

lambda = 7;
maxEpiDist = 15;
maxDist = 150;


local_setup

for i = 1:length(imgNrVec)
    imgNr = imgNrVec(i);
img0 = imread([kitti_path '/00/image_0/' ...
sprintf('%06d.png',imgNr)]);
img1 = imread([kitti_path '/00/image_0/' ...
sprintf('%06d.png',imgNr+1)]);

[keypoints0, descriptors0] = getHarrisFeatures(img0);
[keypoints1, descriptors1] = getHarrisFeatures(img1);

matches = matchDescriptors(descriptors0, descriptors1, lambda);
matchesLoc = matchDescriptorsLocally(descriptors0, descriptors1, keypoints0, keypoints1, lambda, maxDist);
matchesAck = matchDescriptorsAckermannConstrained(descriptors0, descriptors1, keypoints0, keypoints1, lambda, K, maxEpiDist,maxDist);

[~, indx0, indx1] = find(matches);
[~, indx0Loc, indx1Loc] = find(matchesLoc);
[~, indx0Ack, indx1Ack] = find(matchesAck);

[~,inliers] = estimateEssentialMatrix_RANSAC(homogenize2D(keypoints0(:,indx0)),homogenize2D(keypoints1(:,indx1)),K,K);
[~,inliersLoc] = estimateEssentialMatrix_RANSAC(homogenize2D(keypoints0(:,indx0Loc)),homogenize2D(keypoints1(:,indx1Loc)),K,K);
[~,inliersAck] = estimateEssentialMatrix_RANSAC(homogenize2D(keypoints0(:,indx0Ack)),homogenize2D(keypoints1(:,indx1Ack)),K,K);

fh = figure(2);clf;
set(fh,'Color','white','Position', get(0,'Screensize'));
subplot(3,1,1);
imshow(img1);
hold on;
plotMatchVectors(keypoints0(:,indx0),keypoints1(:,indx1),'r');
plotMatchVectors(keypoints0(:,indx0(inliers)),keypoints1(:,indx1(inliers)),'g');
title({'Unconstrained matching';sprintf('#tracked: %i, #inliers: %i, ratio: %f',nnz(matches),nnz(inliers),nnz(inliers)/nnz(matches))})

subplot(3,1,2)
imshow(img1);
hold on
plotMatchVectors(keypoints0(:,indx0Loc),keypoints1(:,indx1Loc),'r');
plotMatchVectors(keypoints0(:,indx0Loc(inliersLoc)),keypoints1(:,indx1Loc(inliersLoc)),'g');
title({'Distance constrained matching';sprintf('#tracked: %i, #inliers: %i, ratio: %f',nnz(matchesLoc),nnz(inliersLoc),nnz(inliersLoc)/nnz(matchesLoc))})

subplot(3,1,3)
imshow(img1);
hold on
plotMatchVectors(keypoints0(:,indx0Ack),keypoints1(:,indx1Ack),'r');
plotMatchVectors(keypoints0(:,indx0Ack(inliersAck)),keypoints1(:,indx1Ack(inliersAck)),'g');
title({'Ackermann constrained matching';sprintf('#tracked: %i, #inliers: %i, ratio: %f',nnz(matchesAck),nnz(inliersAck),nnz(inliersAck)/nnz(matchesAck))});

suptitle('Matching done on same keypoints with different constraints')


fh.PaperPositionMode = 'auto';
fig_pos = fh.PaperPosition;
fh.PaperSize = [fig_pos(3) fig_pos(4)];
print(fh,['other_data\matchComp',num2str(imgNr)],'-depsc')
end