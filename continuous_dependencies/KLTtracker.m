function [transform, translation, valid] = KLTtracker(img, transform, img_templates, match_threshold)

num_templates = size(transform,2);
assert(size(transform,1)==6);
assert(size(img_templates,3)==num_templates);

% Flip it! This is to comply with LK-tracker from the exercise sessions
transform(5:6,:) = flipud(transform(5:6,:));

debug = true;

if debug
    prev_pos = transform(5:6,:);
end

translation = zeros(2,num_templates);
valid = zeros(1,num_templates);

%for k = 1:num_templates
%    [transform(:,k), translation(:,k), valid(k)] = KLTracker_loop(img, transform(:,k), img_templates(:,:,k), match_threshold);
%end

[transform, translation, valid] = KLTtracker_loop(img, transform, img_templates, match_threshold);

if debug
    curr_pos = translation;

    figure(10);clf(10);
    imshow(img);hold on; % Show the movie frame
    title(['Tracker']) % Display the frame number on top of the image
    set(gcf,'Color','w'); axis on,

    diff = curr_pos - prev_pos;
    plotMatchVectors(flipud(prev_pos),flipud(curr_pos));
    drawnow
    hold off
end


% Flip back! To conform with the standard in our VO.
transform(5:6,:) = flipud(transform(5:6,:));
translation = flipud(translation);
valid = valid > 0;

end