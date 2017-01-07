function [thetaEst, varTheta, inliers] = onePointHistogramVote(p1, p2, nBins, K, maxDist)
debug = false; debugFig = 1;

if size(p1,1) == 2
    p1 = homogenize2D(p1);
end
if size(p2,1) == 2
    p2 = homogenize2D(p2);
end

R = [0, 0, 1;
        -1, 0, 0;
        0, -1, 0];
    

p1 = K\p1;
p2 = K\p2;

p1 = R*p1;
p2 = R*p2;


p1 = normc(p1);
p2 = normc(p2);

% binSpread = pi/3;
% binStep = binSpread/nBins;
% binEdges = (0:nBins)*binStep - binSpread/2;


theta = -2*atan((p2(2,:).*p1(3,:) - p1(2,:).*p2(3,:))./(p2(1,:).*p1(3,:) + p1(1,:).*p2(3,:)));
meanTheta = mean(theta);
varTheta = var(theta); % if theta is too spread, this could be used to tell
%that this is going to fail.

binSpread = 4*sqrt(varTheta);
binStep = binSpread/nBins;
binEdges = ((0:nBins) - nBins/2)*binStep + meanTheta;
thetaCount = histcounts(theta, binEdges);

if debug
    figure(debugFig);clf;
    stairs(binEdges(1:end-1)*180/pi,thetaCount);
end

[thetaCountMax, maxCountInd] = max(thetaCount);
if maxCountInd > 1 && maxCountInd < nBins 
    y = thetaCount(maxCountInd + (-1:1));
    x = binEdges(maxCountInd + (-1:1)) + binStep/2;
    n = 2;
    p = polyfit(x,y,n); %y = p(1)*x^2 + p(2)*x + p(3)
    thetaEst = -p(2)/(2*p(1)); % dy/dx = 2p(1)x + p(2) = 0
    x = thetaEst;
    y = p(1)*x^2 + p(2)*x + p(3);
    if debug
        hold on;
        scatter(thetaEst*180/pi,y);
    end
else
    thetaEst = binEdges(maxCountInd) + binStep/2; %middle of the bin
    if debug
        hold on
        scatter(thetaEst*180/pi, thetaCountMax);
    end
end

if debug
    fprintf('1point estimates %f deg\n', thetaEst*180/pi)
end

if nargout == 3
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

