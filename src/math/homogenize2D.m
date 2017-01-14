function p_hom = homogenize2D(p)
    assert(size(p,1)==2);
    N = size(p,2);
    p_hom = [p ; ones(1,N)];
end