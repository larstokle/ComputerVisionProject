function [thetaEst, inliers] = onePointHistogramVote(p1, p2, nBins, K, maxDist)
debug = true; debugFig = 1;

nEdges = nBins+1;
binEdges = (0:nEdges)*pi/nEdges - pi/2;

theta = -2*atan((p2(2,:)-p1(2,:))./(p2(1,:) + p1(1,:)));
%varTheta = var(theta); % theta is too spread, this could be used to tell
%that this is going to fail.

thetaCount = histcounts(theta, binEdges);

if debug
    figure(debugFig);clf;
    histogram(theta);
end

[thetaCountMax, maxCountInd] = max(thetaCount);
if maxCountInd > 1 && maxCountInd < nEdges
    y = thetaCount(maxCountInd + (-1:1));
    x = binEdges(maxCountInd + (-1:1)) + pi/(2*nEdges);
    n = 2;
    p = polyfit(x,y,n); %y = p(1)*x^2 + p(2)*x + p(3)
    thetaEst = -p(2)/(2*p(1)); % dy/dx = 2p(1)x + p(2) = 0
    x = thetaEst;
    y = p(1)*x^2 + p(2)*x + p(3);
    if debug
        hold on;
        scatter(thetaEst,y);
    end
else
    thetaEst = nEdges(maxCountInd) + pi/(2*nEdges); %middle of the bin
    if debug
        hold on
        scatter(thetaEst, thetaCountMax);
    end
end


R = [0, 0, 1;
    -1, 0, 0;
    0, -1, 0];
E = [0, 0, sin(thetaEst/2);
     0, 0, cos(thetaEst/2);
     sin(thetaEst/2), -cos(thetaEst/2), 0];
E = R'*E*R;
 
F = inv(K')*E*inv(K);
p1_hom = [p1;ones(1,size(p1,2))];
p2_hom = [p2;ones(1,size(p2,2))];

dists = distPoint2EpipolarLine_pointwise(F,p2_hom,p1_hom);
inliers = dists < maxDist;
end

% function dist = distPoint2EpipolarLine_pointwise(F,p1,p2)
% 
%     NumPoints = size(p1,2);
%     homog_points = [p1, p2];
%     epi_lines = [F.'*p2, F*p1];
%     denom = sum(epi_lines(1:2,:).^2, 1);
%     dist = (sum(epi_lines.*homog_points,1).^2)./denom ;
%     dist = sqrt(dist(1:NumPoints) + dist(NumPoints+1:end));
%     
% %quiver(
% %       [p1(1,find(inlier)), p2(1,find(inlier))],...
% %       [p1(2,find(inlier)), p2(2,find(inlier))],...
% %       [epi_lines(1,find(inlier))./denom(find(inlier)), epi_lines(1,NumPoints + find(inlier))./denom(NumPoints + find(inlier))],...
% %       [epi_lines(2,NumPoints + find(inlier))./denom(find(inlier)), epi_lines(2,NumPoints + find(inlier))./denom(NumPoints + find(inlier))]...
% %       )
% end

