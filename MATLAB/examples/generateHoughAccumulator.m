function hough_img = generateHoughAccumulator(img, theta_num_bins, rho_num_bins)
[y,x] = size(img);
% init the accumulator
acc = zeros(rho_num_bins, theta_num_bins);
% get the theta step for each bin
theta_step = pi/theta_num_bins;

% Run through each pixel in the image
for i = 1:y
    for j = 1:x
        if img(i,j) > 0 %The pixel is part of an edge.
            theta = 0;
            %already_inc = zeros(0,3);
            for k = 1:theta_num_bins % For each x entry in the accumulator
                % Parameterize in Polar space.
                hough_x = k;
                hough_y = (rho_num_bins/2) - round(-j*sin(theta)+i*cos(theta));
                
                % This was an attempt at the 3x3 incrementer.
                % Ended up getting things working quite well without it.
%                 % Create the list of values to increment
%                 val = 0.33;
%                 new_inc = [...
%                     [hough_x-1,hough_y-1,val];...
%                     [hough_x,hough_y-1,val];...
%                     [hough_x+1,hough_y-1,val];...
%                     [hough_x-1,hough_y,val];...
%                     [hough_x,hough_y,1];...
%                     [hough_x+1,hough_y,val];...
%                     [hough_x-1,hough_y+1,val];...
%                     [hough_x,hough_y+1,val];...
%                     [hough_x+1,hough_y+1,val];...
%                     ];
%                 
%                 for m = 1:size(new_inc)
%                     found = 0;
%                     for n = 1:size(already_inc)
%                         if already_inc(n,1) == new_inc(m,1)...
%                         && already_inc(n,2) == new_inc(m,2)
%                             found = 1;
%                             break;
%                         end
%                     end
%                     
%                     if found == 0 ...
%                     && new_inc(m,1) > 0 ...
%                     && new_inc(m,1) < theta_num_bins ...
%                     && new_inc(m,2) > 0 ...
%                     && new_inc(m,2) < rho_num_bins ...
%                     
%                         acc(new_inc(m,2),new_inc(m,1)) = acc(new_inc(m,2),new_inc(m,1)) + new_inc(m,3);
%                     end
%                     
%                 end
%                 already_inc = new_inc;
                
                % Increment the value in the accumulator.
                acc(hough_y,hough_x) = acc(hough_y,hough_x) + 1;
                %sassignin("base","new_inc",new_inc)
                %fprintf("X:%.2f Y: %.2f\n", hough_x, hough_y)
                
                theta = theta + theta_step;
                
            end
        end
    end
end

% To write the accumulator as an image, scale all values to be
% between 0 and 255. Max value in accumulator==255
white_balance = 255/max(max(acc));
for i = 1:rho_num_bins
    for j = 1:theta_num_bins
        % Scale value and round to int.
        acc(i,j) = round(acc(i,j) * white_balance);
    end
end

% Convert accumulator to int matrix and return.
hough_img = uint8(acc);
%assignin("base","acc",acc)
%fprintf("Width: %d, Height: %d\n",width,height)