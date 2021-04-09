function houghAccumulator()

output_folder_name = '../processed_images/hough';

% Create the output folder if it doesn't exist
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end
input_folder_name = '../processed_images/edges';


% Read the image.
input_img_name = strcat(input_folder_name,'/image-1.png');
img = imread(input_img_name);


[y,x] = size(img);
acc_size = x;
radius = 25;
% init the accumulator
acc = zeros(acc_size, acc_size);

% Run through each pixel in the image
for i = 1:y
    for j = 1:x
        if img(i,j) > 0 %The pixel is part of an edge.
            for t = 1:360 % Draw each circle in the accumulator space
                a = round(i-radius*sin(t*pi/180));
                b = round(j-radius*cos(t*pi/180));
                % Gotta be within the range
                if a > 0 && a <= acc_size && b > 0 && b<=acc_size
                    % Increment the value in the accumulator.
                    acc(a,b) = acc(a,b) + 1; 
                end
            end
        end
    end
end

% To write the accumulator as an image, scale all values to be
% between 0 and 255. Max value in accumulator==255
white_balance = 255/max(max(acc));
for i = 1:acc_size
    for j = 1:acc_size
        % Scale value and round to int.
        acc(i,j) = round(acc(i,j) * white_balance);
    end
end

% Write accumulator as an image
output_img_name = strcat(output_folder_name,'/image-1.png');
imwrite(uint8(acc), output_img_name);

%Threshold at which we consider a point the center of a circle
threshold = 200;

[hough_y, hough_x] = size(acc);

% Find all centers that pass threshold
% Store each line as a,b
circles = zeros(0,2);
for i = 1:hough_y
    for j = 1:hough_x
        if acc(i,j) >= threshold
            circles = [circles; [j,i]];
        end
    end 
end


% Create figure and show the image so that we can draw lines on top of it.
fh = figure();
figure(fh);
imshow(imread('../images/image-1.png'));
hold on;
th = 0:pi/50:2*pi;
for i = 1:size(circles)
    xunit = radius * cos(th) + circles(i,1);
    yunit = radius * sin(th) + circles(i,2);
    plot(xunit, yunit,'r-','LineWidth',3);
end
hold off;
% Save the image with the lines drawn on it.
circle_detected_img = saveAnnotatedImg(fh);
delete(fh);

imwrite(uint8(circle_detected_img),'../output/bloodcells/image-1.png');

% fh = figure();
% subplot(2,2,1)
% imshow(imread('../images/image-1.png'));
% subplot(2,2,2)
% imshow(imread('../processed_images/edges/image-1.png'));
% subplot(2,2,3)
% imshow(im2double(uint8(acc)));
% subplot(2,2,4)
% imshow(circle_detected_img);

end

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
end

