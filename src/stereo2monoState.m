function state = stereo2monoState(stereoState,K)

state.poses = stereoState.poses;
state.landmark_keypoints = stereoState.keypoints;
state.landmark_descriptors = stereoState.descriptors;
state.landmark_transforms = [zeros(4,size(state.landmark_keypoints,2)); state.landmark_keypoints];
state.landmarks = [stereoState.landmarks;ones(1,size(stereoState.landmarks,2))];

state.candidate_keypoints = stereoState.keypoints;
state.candidate_descriptors = stereoState.descriptors;
state.candidate_bearings_1 = calculateBearingVectors(stereoState.keypoints,reshape(state.poses(:,end),4,4),K);
state.candidate_keypoints_1 = stereoState.keypoints;
state.candidate_pose_idx_1 = size(state.poses,2)*ones(1,size(stereoState.keypoints,2));
end