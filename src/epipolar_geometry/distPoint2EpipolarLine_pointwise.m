% distPoint2EpipolarLine  Compute the point-to-epipolar-line distance
%
%   Input:
%   - F(3,3): Fundamental matrix
%   - p1(3,NumPoints): homogeneous coords of the observed points in image 1
%   - p2(3,NumPoints): homogeneous coords of the observed points in image 2
%
%   Output:
%   - cost: sum of squared distance from points to epipolar lines
%           normalized by the number of point coordinates


function dist = distPoint2EpipolarLine_pointwise(F,p1,p2)

    NumPoints = size(p1,2);
    homog_points = [p1, p2];
    epi_lines = [F.'*p2, F*p1];
    denom = epi_lines(1,:).^2 + epi_lines(2,:).^2;
    dist = (sum(epi_lines.*homog_points,1).^2)./denom ;
    dist = sqrt(dist(1:NumPoints) + dist(NumPoints+1:end));
    
end

function dist = testDist2EpipolarLine(F,p1,p2)

    N = length(p1);    
    dist = zeros(N,1);

    for i = 1:N
       dist(i) = distPoint2EpipolarLine(F,p1(:,i),p2(:,i)); 
    end
end
