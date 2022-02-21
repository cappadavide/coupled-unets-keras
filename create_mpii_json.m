newJSON.filepath = {};
nImgs = 1;
for i=1:length(annotations)
    if isfield(annotations(i).annorect,'annopoints')==1
        if i==1
            disp("Sono entrato uao");
        end
        newJSON.filepath = [newJSON.filepath;annotations(i).image.name];
        for k=1:length(annotations(i).annorect)
            if isfield(annotations(i).annorect(k).annopoints,'point')==1
                tempTable = struct2table(annotations(i).annorect(k).annopoints.point);
                sortedPoints = sortrows(tempTable,'id','ascend');
                sortedPoints = table2struct(sortedPoints);
                newJSON.keypoints(nImgs).people(k).points = repmat(struct('x',0,'y',0,'id',-1),1,16);
                count = 1;
                j=1;
                while count<=16
                   if j<=length(sortedPoints)
                       if sortedPoints(j).id == count-1
                           newJSON.keypoints(nImgs).people(k).points(count).x = sortedPoints(j).x;
                           newJSON.keypoints(nImgs).people(k).points(count).y = sortedPoints(j).y;
                           j = j + 1;
                       end
                   end
                   newJSON.keypoints(nImgs).people(k).points(count).id = count-1;
                   count = count + 1;
                end
            end
            
        end
        nImgs = nImgs + 1;
    end
end
