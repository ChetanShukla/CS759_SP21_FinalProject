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
acc_size = 200*2;
radius = 50;
% init the accumulator
acc = zeros(acc_size, acc_size);

% Run through each pixel in the image
for i = 1:y
    for j = 1:x
        if img(i,j) > 0 %The pixel is part of an edge.
            for t = 1:360 % For each x entry in the accumulator
                % Parameterize in Polar space.
                b = round(i-radius*sin(t*pi/180))+radius;
                a = round(j-radius*cos(t*pi/180))+radius;
                
                % Increment the value in the accumulator.
                acc(a,b) = acc(a,b) + 1;     
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


output_img_name = strcat(output_folder_name,'/image-1.png');


