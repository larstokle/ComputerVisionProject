function [ps, ds, valid] = KLTtracker(img, ps, img_templates, match_threshold)
height = 3;

imgPyr = cell(height,1);
imgTempPyr = cell(height,1);

imgPyr{1} = img;
imgTempPyr{1} = img_templates;

for i = 2:height
    imgPyr{i} = impyramid(imgPyr{i-1}, 'reduce');
    imgTempPyr{i} = impyramid(imgTempPyr{i-1}, 'reduce');
end

ps(5:6,:) = ps(5:6,:)/2^height;
transOpts = struct('TranslationIterations',6,'AffineIterations',0,...
        'TolP',1e-5,'Sigma',1.5);
affineOpts = struct('TranslationIterations',0,'AffineIterations',5,...
    'TolP',1e-5,'Sigma',1.5);
ds = zeros(2,size(ps,2));
T_error = zeros(1,size(ps,2));
for k = 1:size(ps,2)
    %do translation pyramidal
    for i = height:-1:1
        ps(5:6,k) = ps(5:6,k)*2;
        ps(:,k) = LucasKanadeAffine(double(imgPyr{i}), ps(:,k),double(imgTempPyr{i}(:,:,k)), transOpts);
    end
    %find affine transform to see template match
    ds(:,k) = ps(5:6,k);
    [ps(:,k), ~, T_error(k)] = LucasKanadeAffine(double(img), ps(:,k), double(img_templates(:,:,k)), affineOpts);
end

T_error = T_error/(255^2*numel(img_templates(:,:,1))); %scale to fraction of maximum

valid = T_error < match_threshold; %good match
valid = valid & ds(1,:) > 0 & ds(2,:) > 0; %inside top and left
valid = valid & ds(1,:) <= size(img,2) & ds(2,:) <= size(img,1);
end