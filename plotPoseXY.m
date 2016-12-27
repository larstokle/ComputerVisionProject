function plotPoseXY(axHandle, pose)
%plotPose(axHandle, pose): plots a coordinate system in the XY-plane. x,y,z = r,g,b
%
%   Input:
%       axHandle:   handle to axes to plot in
%       pose:       (4x4) homogenous transformation matrix to plot

quiver(axHandle, pose(1,4),pose(2,4),pose(1,1),pose(2,1),'Color','r'); %camera0 x-axis in "World"
quiver(axHandle, pose(1,4),pose(2,4),pose(1,2),pose(2,2),'Color','g'); %camera0 y-axis in "World"
quiver(axHandle, pose(1,4),pose(2,4),pose(1,3),pose(2,3),'Color','b'); %camera0 z-axis in "World"
end