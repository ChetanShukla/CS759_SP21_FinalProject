function circles = findCircles(acc)

    [sorted_vals,ix] = sort(acc(:),'descend');
    threshold = sorted_vals(1)/2;
    counter = 1;
    circles = zeros(0,3);
    out_range = true;
    while(sorted_vals(counter)>threshold)
        [i,j,k] = ind2sub(size(acc),ix(counter));
        
        for z = 1:size(circles,1)
            circle = circles(z,:);
            distance = sqrt((circle(1)-j)^2+(circle(2)-i)^2);
            if  distance < 10
                out_range = false;
                break
            end
        end
        
        if out_range
           circles = [circles; [j,i,k]];
        end
        
        out_range = true;
        counter = counter +1;
    end
end