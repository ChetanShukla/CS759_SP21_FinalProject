function circles = findCircles(acc,shrink_factor,radius_start)

    [sorted_vals,ix] = sort(acc(:),'descend');
    threshold = sorted_vals(1)/2;
    counter = 1;
    val = 10;
    circles = zeros(0,3);
    while(sorted_vals(counter)>threshold)
        [i,j,k] = ind2sub(size(acc),ix(counter));
        
        if acc(i,j,k) > 0
            circles = [circles; [j*shrink_factor-1,i*shrink_factor-1,(k+radius_start)*shrink_factor]];
            acc(clamp(i-val):clamp(i+val),clamp(j-val):clamp(j+val),:) = 0;
        end
        
        counter = counter +1;
    end
end

function ret_val = clamp(val)
    ret_val = min(max(val,1),64);
end 