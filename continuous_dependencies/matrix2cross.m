function x = matrix2cross(M)
% x = matrix2cross(M)
%
%       M =[0       ,-x(3)  ,x(2)   ; 
%           x(3)    ,0      ,-x(1)  ;
%           -x(2)   ,x(1)   ,0      ];
%   
%       x = [M(3,2); M(1,3); M(2,1)];
%
%   see also cross2matrix

    x = [M(3,2); M(1,3); M(2,1)];
end