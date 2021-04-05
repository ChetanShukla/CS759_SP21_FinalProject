function runPreprocess

% % Settings to make sure images are displayed without borders.
% orig_imsetting = iptgetpref('ImshowBorder');
% iptsetpref('ImshowBorder', 'tight');
% temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

color_to_blackwhite
image_segmentation
edge_detection

% Read in each image, convert to black and white, and write the image back out.
function color_to_blackwhite()

output_folder_name = '../processed_images/black_white';

% Create the output folder if it doesn't exist
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end

input_folder_name = '../images';
for i = int16(1):int16(100)

    %R ead the image
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

end


% Find the edges in the image.
% Only used to help with concurrent devleopment. Will only use output of GPU
% edge detection once completed.
function edge_detection()

output_folder_name = '../processed_images/edges';

% Create the output folder if it doesn't exist
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end
input_folder_name = '../processed_images/black_white';
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
end

    
    
    
    
    
    
    


