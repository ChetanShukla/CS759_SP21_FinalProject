function runPreprocess

% % Settings to make sure images are displayed without borders.
% orig_imsetting = iptgetpref('ImshowBorder');
% iptsetpref('ImshowBorder', 'tight');
% temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

color_to_blackwhite
edge_detection


function color_to_blackwhite()

output_folder_name = '../processed_images/black_white';
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end
input_folder_name = '../images';
for i = int16(1):int16(100)
    input_img_name = strcat(input_folder_name,'/image-', num2str(i), '.png');
    img = imread(input_img_name);

    % Convert the image to grayscale
    gray_img = rgb2gray(img);
    
    output_img_name = strcat(output_folder_name,'/image-', num2str(i), '.png');
    imwrite(gray_img, output_img_name);
end


function edge_detection()

output_folder_name = '../processed_images/edges';
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end
input_folder_name = '../processed_images/black_white';
for i = int16(1):int16(100)
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

    
    
    
    
    
    
    


