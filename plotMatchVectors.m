function plotMatchVectors(p1,p2)
    % P2 is query kp - i.e new image
    % P1 is db  kp - i.e old image
    
    dim = size(p1,1);
    
    assert(numel(p1)==numel(p2))
    assert(dim <= 3 && dim >=2);
    assert(all(p1(3,:)==1));
    assert(all(p2(3,:)==1));
    
    if dim == 3
       % De-homogenize
       p1 = p1(1:2,:);
       p2 = p2(1:2,:);
    end
    
    p1 = flipud(p1);
    p2 = flipud(p2);
    
    x_from = p2(1,:); % Query kp
    x_to = p1(1,:); % DB kp
    y_from = p2(2,:); % Query kp
    y_to = p1(2,:); % DB kp
    plot([y_from; y_to], [x_from; x_to], 'g-', 'Linewidth', 3)
end