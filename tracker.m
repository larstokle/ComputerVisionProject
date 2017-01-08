function state = tracker(image,prev_state)
    
    %Unwrap state
    prev_keypoints = prev_state.landmark_keypoints;
    prev_descriptors = prev_state.landmark_descriptors;
    prev_landmarks = prev_state.landmarks;
    
    prev_keypoints = flipud(prev_keypoints);

    [keypoints, descriptors, idx_of_tracked] = harrisTracker(prev_keypoints,prev_descriptors,image);
    
    keypoints = flipud(keypoints);
    
    N = size(keypoints,2);
    
    disp(['Num tracked: ' num2str(N)]);
       
    % Set new state
    state.landmark_keypoints = keypoints;
    state.landmark_descriptors = descriptors;
    state.landmarks = prev_landmarks(:,idx_of_tracked);
    
    assert(size(keypoints,2)==N);
    assert(size(descriptors,2)==N);
    assert(size(state.landmarks,2)==N);
end