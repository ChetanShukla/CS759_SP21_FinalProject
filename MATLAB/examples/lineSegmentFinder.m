function cropped_line_img = lineSegmentFinder(orig_img, hough_img, hough_threshold, edge_img)
threshold = hough_threshold;
%threshold = 255 - 3 * mean(mean(hough_img(hough_img>0)));

[hough_y, hough_x] = size(hough_img);

% Store each line as theta,rho
lines = zeros(0,2);

theta_step = pi/hough_x;
%Find all the lines
for i = 1:hough_y
    theta = 0;
    for j = 1:hough_x
        if hough_img(i,j) >= threshold
            %fprintf("X: %d, Y: %d, Value: %d, Threshold: %d\n",j,i,hough_img(i,j),threshold);
            lines = [lines; [theta, hough_y/2 - i]];
        end
        theta = theta + theta_step;
    end 
end
%assignin("base","lines",lines)

lines = sortrows(lines,[2,1]);

%Average lines that are basically the same.
lines_filtered = zeros(0,2);

prev_line = lines(1,:);
sum_theta = prev_line(1);
sum_rho = prev_line(2);
counter = 1;
for i = 2:size(lines)
    cur_line = lines(i,:);
    if prev_line(2) <= cur_line(2) && cur_line(2) <= prev_line(2)+4 && ...
       prev_line(1) <= cur_line(1) && cur_line(1) <= prev_line(1) + theta_step*2
   
        sum_theta = sum_theta + cur_line(1);
        sum_rho = sum_rho + cur_line(2);
        counter = counter + 1;
        prev_line = cur_line;
    else
        lines_filtered = [lines_filtered; [sum_theta/counter, sum_rho/counter]];
        sum_theta = cur_line(1);
        sum_rho = cur_line(2);
        counter = 1;
        prev_line = cur_line;
    end
    % Ensure the last line/lines average gets added
    if i == size(lines,1)
        lines_filtered = [lines_filtered; [sum_theta/counter, sum_rho/counter]];
    end
end

lines = lines_filtered;

[orig_y,orig_x] = size(orig_img);

fh = figure();
figure(fh);
imshow(orig_img);
for i = 1:size(lines)
    theta = lines(i,1);
    rho = lines(i,2);
    
    tolerance = 3;
    
    % Calculate the slope of the line to determine which alg to use.
    % x = 0,1. y2-y1/x2-x1
    slope = abs((sec(theta) * (rho + 1 * sin(theta)))-(sec(theta) * (rho + 0 * sin(theta))));
    
    % Vary x or y based on the slope. Vertical lines are hard to detect
    % when varing x.
    if slope > 1
        
        %Find the start of the line
        y_start = 1;
        found = 0;
        for j = 1:orig_y
            x = round(j*cot(theta) - rho*csc(theta));

            for k = x-tolerance:x+tolerance
                if 0 < k && k <= orig_x
                    if edge_img(j,k) > 0
                        y_start = j;
                        found = 1;
                        break;
                    end
                end
            end

            if found == 1
                break;
            end
        end

        %Find the end of the line
        y_end = y_start;
        found = 0;
        for j = orig_y:-1:y_start
            x = round(j*cot(theta) - rho*csc(theta));

            for k = x-tolerance:x+tolerance
                if 0 < k && k <= orig_x
                    if edge_img(j,k) > 0
                        y_end = j;
                        found = 1;
                        break;
                    end
                end
            end

            if found == 1
                break;
            end
        end
    
        y = linspace(y_start,y_end);
        x = y*cot(theta) - rho*csc(theta);

    else
    
        %Find the start of the line
        x_start = 1;
        found = 0;
        for j = 1:orig_x
            y = round(sec(theta) * (rho + j * sin(theta)));

            for k = y-tolerance:y+tolerance
                if 0 < k && k <= orig_y
                    if edge_img(k, j) > 0
                        x_start = j;
                        found = 1;
                        break;
                    end
                end
            end

            if found == 1
                break;
            end
        end

        %Find the end of the line
        x_end = x_start;
        found = 0;
        for j = orig_x:-1:x_start
            y = round(sec(theta) * (rho + j * sin(theta)));

            for k = y-tolerance:y+tolerance
                if 0 < k && k <= orig_y
                    if edge_img(k, j) > 0
                        x_end = j;
                        found = 1;
                        break;
                    end
                end
            end

            if found == 1
                break;
            end
        end

        x = linspace(x_start,x_end);
        y = sec(theta) * (rho + x * sin(theta));
    end
    
    line(x,y,'Color','red');
    %fprintf("Theta: %d Rho: %d\n",lines(i,1), lines(i,2))
end

cropped_line_img = saveAnnotatedImg(fh);

delete(fh);

function annotated_img = saveAnnotatedImg(fh)
figure(fh); % Shift the focus back to the figure fh

% The figure needs to be undocked
set(fh, 'WindowStyle', 'normal');

% The following two lines just to make the figure true size to the
% displayed image. The reason will become clear later.
img = getimage(fh);
truesize(fh, [size(img, 1), size(img, 2)]);

% getframe does a screen capture of the figure window, as a result, the
% displayed figure has to be in true size. 
frame = getframe(fh);
frame = getframe(fh);
pause(0.5); 
% Because getframe tries to perform a screen capture. it somehow 
% has some platform depend issues. we should calling
% getframe twice in a row and adding a pause afterwards make getframe work
% as expected. This is just a walkaround. 
annotated_img = frame.cdata;