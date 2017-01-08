function point = disparityToPoint(...
    disp, K, baseline, coord)
% returnes the world point given imagepoint, disparity, baseline and
% kalibration matrix;
% point should be 3x1

assert(disp > 0);

px_left = [coord; 1];

px_right = px_left;
px_right(2) = px_right(2) - disp;

bv_left = K^-1 * px_left;
bv_right = K^-1 * px_right;

b = [baseline; 0; 0];
A = [bv_left, -bv_right];
x = (A' * A) \ (A' * b);
point = bv_left * x(1);

end

