function [thetaEst, inliers] = onePointHistogramVote(p1, p2, nBins, K, maxDist)
debug = true; debugFig = 1;

nEdges = nBins+1;
binEdges = (0:nEdges)*pi/nEdges - pi/2;

theta = -2*atan((p2(2,:)-p1(2,:))./(p2(1,:) + p2(1,:)));
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



E = [0, 0, sin(thetaEst/2);
     0, 0, cos(thetaEst/2);
     sin(thetaEst/2), -cos(thetaEst/2), 0];
 
F = K'\E/K;

dists = distPoint2EpipolarLine_pointwise(F,[p1;ones(1,size(p1,2))],[p2;ones(1,size(p2,2))]);
inliers = dists < maxDist;
end

