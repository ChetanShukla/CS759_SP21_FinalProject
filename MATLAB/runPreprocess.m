function runPreprocess

% Settings to make sure images are displayed without borders.
orig_imsetting = iptgetpref('ImshowBorder');
iptsetpref('ImshowBorder', 'tight');
temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

output_folder_name = 'output';
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end


%--------------------------------------------------------------------------
% Tests for Challenge 1: Hough transform
%--------------------------------------------------------------------------
%%
function runEdgeDetection()

img_list = {'hough_1', 'hough_2', 'hough_3'};
%fh = figure;
for i = 1:length(img_list)
    img = imread(['input/' img_list{i} '.png']);
    
    % Canny edge detection
    thresh = 0.1;
    edge_img = edge(img,'canny', thresh);
    
    
    %subplot(2, 2, i);
    %imshow(edge_img);
    
    
    % Note: The output from edge is an image of logical type.
    % Here we cast it to double before saving it.
    imwrite(im2double(edge_img), ['output/edge_' img_list{i} '.png']);
end

