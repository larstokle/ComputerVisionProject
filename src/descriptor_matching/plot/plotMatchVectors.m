function plotMatchVectors(p1,p2,lineSpec,ax)
    % P2 is query kp - i.e new image
    % P1 is db  kp - i.e old image
    if ~exist('lineSpec','var') || isempty(lineSpec)
        lineSpec='g-';
    end
    
    if ~exist('ax','var') || isempty(ax)
        ax = gca
    end
    
    dim = size(p1,1);
    
    assert(numel(p1)==numel(p2))
    assert(dim <= 3 && dim >=2);
    
    if dim == 3
       % De-homogenize
       assert(all(p1(3,:)==1));
       assert(all(p2(3,:)==1));
       p1 = p1(1:2,:);
       p2 = p2(1:2,:);
    end
    
    p1 = flipud(p1);
    p2 = flipud(p2);
    
    x_from = p2(1,:); % Query kp
    x_to = p1(1,:); % DB kp
    y_from = p2(2,:); % Query kp
    y_to = p1(2,:); % DB kp
    quiver(y_from,x_from,y_to-y_from,x_to-x_from,0, lineSpec, 'Linewidth', 2,'Parent',ax)
end