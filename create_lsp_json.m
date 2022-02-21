 newJSON.filepath = {};
for i=1:2000
    newJSON.filepath = [newJSON.filepath;sprintf('img%04d.jpg',i)];
    tmpCellArray = joints(:,:,i);
    newJSON.keypoints(i).points = repmat(struct('x',0,'y',0,'id',-1),1,14);
    for j=1:14
        newJSON.keypoints(i).points(j).x = tmpCellArray(1,j);
        newJSON.keypoints(i).points(j).y = tmpCellArray(2,j);
        newJSON.keypoints(i).points(j).id = j;
    end
end