function plotPoseXY(axHandle, pose)
%plotPose(axHandle, pose): plots a coordinate system in the XY-plane. x,y,z = r,g,b
%
%   Input:
%       axHandle:   handle to axes to plot in
%       pose:       (4x4) homogenous transformation matrix to plot

quiver(axHandle, pose(3,4),-pose(1,4),pose(3,1),-pose(1,1),'Color','r'); %camera0 x-axis in "World"
quiver(axHandle, pose(3,4),-pose(1,4),pose(3,2),-pose(1,2),'Color','g'); %camera0 y-axis in "World"
quiver(axHandle, pose(3,4),-pose(1,4),pose(3,3),-pose(1,3),'Color','b'); %camera0 z-axis in "World"
end