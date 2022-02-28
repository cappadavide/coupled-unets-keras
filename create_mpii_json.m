nImgs = 1;
annotations = RELEASE.annolist;
%28883
for i=1:length(annotations)
    if isfield(annotations(i).annorect,'annopoints')==1
        for k=1:length(annotations(i).annorect) % persone
            if isempty(annotations(i).annorect(k).scale)==0 && isempty(annotations(i).annorect(k).objpos)==0 && isfield(annotations(i).annorect(k).annopoints,'point')==1
                tempTable = struct2table(annotations(i).annorect(k).annopoints.point);
                sortedPoints = sortrows(tempTable,'id','ascend');
                sortedPoints = table2struct(sortedPoints);
                newJSON.people(nImgs).filepath = annotations(i).image.name;
                newJSON.people(nImgs).x1 = annotations(i).annorect(k).x1;
                newJSON.people(nImgs).y1 = annotations(i).annorect(k).y1;
                newJSON.people(nImgs).x2 = annotations(i).annorect(k).x2;      
                newJSON.people(nImgs).y2 = annotations(i).annorect(k).y2;
                newJSON.people(nImgs).keypoints = repmat(struct('x',0,'y',0,'id',-1),1,16);
                newJSON.people(nImgs).scale = annotations(i).annorect(k).scale;
                objpos.x = annotations(i).annorect(k).objpos.x;
                objpos.y = annotations(i).annorect(k).objpos.y;
                newJSON.people(nImgs).objpos = repmat(struct('x',objpos.x,'y',objpos.y),1,1);
                count = 1;
                j=1;
                while count<=16
                   if j<=length(sortedPoints)
                       if sortedPoints(j).id == count-1
                           newJSON.people(nImgs).keypoints(count).x = sortedPoints(j).x;
                           newJSON.people(nImgs).keypoints(count).y = sortedPoints(j).y;
                           j = j + 1;
                       end
                   end
                   newJSON.people(nImgs).keypoints(count).id = count-1;
                   count = count + 1;
                end
                nImgs = nImgs + 1;
            end
        end
    end
end
