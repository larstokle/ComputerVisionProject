function [transform, translation, valid] = KLTtracker_loop(img, transform, img_templates, match_threshold)

num_templates = size(transform,2);
num_template_pixels = numel(img_templates(:,:,1));

assert(size(transform,1)==6);
assert(size(img_templates,3)==num_templates);

height = 1;

image_pyramid = cell(height,1);
template_pyramid = cell(height,1);

image_pyramid{1} = img;
template_pyramid{1} = img_templates;

for i = 2:height
    image_pyramid{i} = impyramid(image_pyramid{i-1}, 'reduce');
    template_pyramid{i} = impyramid(template_pyramid{i-1}, 'reduce');
end

transOpts = struct('TranslationIterations',6,'AffineIterations',0,'TolP',1e-5,'Sigma',1.5);
affineOpts = struct('TranslationIterations',0,'AffineIterations',5,'TolP',1e-5,'Sigma',1.5);
translation = zeros(2,size(transform,2));
error = zeros(1,num_templates);

transform(5:6,:) = transform(5:6,:)/2^height;

for k = 1:num_templates
    %do translation pyramidal
    for i = height:-1:1
        transform(5:6,k) = transform(5:6,k)*2;
        transform(:,k) = LucasKanadeAffine(double(image_pyramid{i}), transform(:,k),double(template_pyramid{i}(:,:,k)), transOpts);
    end
    %find affine transform to see template match
    translation(:,k) = transform(5:6,k);
    [transform(:,k), ~, error(k)] = LucasKanadeAffine(double(img), transform(:,k), double(img_templates(:,:,k)), affineOpts);
end

error = error/(num_template_pixels*255^2); %scale to fraction of maximum

% 0 < (row,col) < size(img)

valid = error < match_threshold; %good match
valid = valid & translation(1,:) > 0 & translation(2,:) > 0; %inside top and left
valid = valid & translation(1,:) <= size(img,1) & translation(2,:) <= size(img,2);


end