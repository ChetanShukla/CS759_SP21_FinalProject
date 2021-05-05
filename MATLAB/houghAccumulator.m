function acc = houghAccumulator(img,shrink_factor,radius_start)

    [y,x] = size(img);
    acc_size = round(x/shrink_factor);
    radius_len = 12/shrink_factor;
    % init the accumulator
    acc = zeros(acc_size, acc_size,radius_len);

    % Run through each pixel in the image
    for i = 1:y
        for j = 1:x
            if img(i,j) > 0 %The pixel is part of an edge.
                for r = 1:radius_len
                    % holder=[];
                    % Lots of duplicate a,b values due to reduced accumulator
                    % mean that we can reduce the number of points we draw. 256
                    % fits nicely into a cuda kernel.
                    % for t = linspace(1,360,256) % Draw each circle in the accumulator space
                    shrunk_i = i/shrink_factor;
                    shrunk_j = j/shrink_factor;
                    for t = 1:360
                        sin_result = sin(t*pi/180);
                        cos_result = cos(t*pi/180);                
                        r_a = radius_start+r;
                        a = round(shrunk_i-r_a*sin_result);
                        b = round(shrunk_j-r_a*cos_result);
                        % holder = [holder; [a,b]];
                        % Gotta be within the range
                        if 0 < a && a <= acc_size && 0 < b && b<=acc_size
                            % Increment the value in the accumulator.
                            acc(a,b,r) = acc(a,b,r) + 1; 
                        end
                    end
    %                 [temp,~,ix] = unique(holder(:,1:2),'rows');
    %                 ix = accumarray(ix,1);
    %                 temp = [temp,ix];
                end
            end
        end
    end

    
end


