function preprocess()

% % Settings to make sure images are displayed without borders.
% orig_imsetting = iptgetpref('ImshowBorder');
% iptsetpref('ImshowBorder', 'tight');
% temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

color_to_blackwhite
image_segmentation
edge_detection


% Read in each image, convert to black and white, and write the image back out.
function color_to_blackwhite()

output_folder_name = '../processed_images/gray';

% Create the output folder if it doesn't exist
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end

input_folder_name = '../images';
for i = int16(1):int16(100)

    % Read the image
    input_img_name = strcat(input_folder_name,'/image-', num2str(i), '.png');
    img = imread(input_img_name);

    % Convert the image to grayscale
    gray_img = rgb2gray(img);
    
    % Write the image
    output_img_name = strcat(output_folder_name,'/image-', num2str(i), '.png');
    imwrite(gray_img, output_img_name);
end

% Run image segmentation to reduce noise during edge detection.
function image_segmentation()

output_folder_name = '../processed_images/segmented';

% Create the output folder if it doesn't exist
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end

input_folder_name = '../processed_images/gray';
for i = int16(1):int16(100)    
    %Read the image
    input_img_name = strcat(input_folder_name,'/image-', num2str(i), '.png');
    img = imread(input_img_name);
    
    % Convert the image to a binary image
    bw_img = imbinarize(img);
    
    % Apply median filtering
    smoothed_img = medfilt2(bw_img);
    
    % Write the image
    output_img_name = strcat(output_folder_name,'/image-', num2str(i), '.png');
    imwrite(smoothed_img, output_img_name);
end


% Find the edges in the image.
% Only used to help with concurrent devleopment. Will only use output of GPU
% edge detection once completed.
function edge_detection()

output_folder_name = '../processed_images/edges/MATLAB';
% Create the output folder if it doesn't exist
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end

binary_output_dir = strcat(output_folder_name,'/binary');
if ~exist(binary_output_dir, 'dir')
   mkdir(binary_output_dir)
end

input_folder_name = '../processed_images/segmented';
for i = int16(1):int16(100)

    % Read the image.
    input_img_name = strcat(input_folder_name,'/image-', num2str(i), '.png');
    img = imread(input_img_name);

    % Canny edge detection
    thresh = 0.2;
    edge_img = edge(img,'canny', thresh);
    
    output_img_name = strcat(output_folder_name,'/image-', num2str(i), '.png');
    
    % Note: The output from edge is an image of logical type.
    % Here we cast it to double before saving it.
    imwrite(im2double(edge_img), output_img_name);
    
    % Create the binary output folder if it doesn't exist
    binary_output_name = strcat(binary_output_dir,'/image-', num2str(i));
    fid = fopen(binary_output_name, 'w');
    fwrite(fid,edge_img);
    fclose(fid);
end

    
    
    
    
    
    
    


