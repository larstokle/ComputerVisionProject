function R = simpleRotX(a)
    R = [1  0       0;
        0   cos(a)  -sin(a);
        0   sin(a)  cos(a)];
end