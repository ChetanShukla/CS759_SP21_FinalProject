function runExamples(varargin)
% runExamples is the "main" interface that lets you execute all the 
% example methods.
%
% Usage:
% runExamples                       : list all the registered functions
% runExamples('function_name')      : execute a specific test
% runExamples('all')                : execute all the registered functions

% Settings to make sure images are displayed without borders.
orig_imsetting = iptgetpref('ImshowBorder');
iptsetpref('ImshowBorder', 'tight');
temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

% Create the output file if it doesn't exist
output_folder_name = 'output';
if ~exist(output_folder_name, 'dir')
   mkdir(output_folder_name)
end

fun_handles = {@runBasicExample, ...
    @runEdgeDetection, @runGenerateHoughAccumulator, @runFindLines, @runFindLineSegments};
% Call test harness
runTests(varargin, fun_handles);

% Basic example to show color to black and white and 
% edge detection all side by side.
function runBasicExample()
basicExample;


function runEdgeDetection()

img_list = {'hough_1', 'hough_2', 'hough_3'};

for i = 1:length(img_list)
    img = imread(['input/' img_list{i} '.png']);
    
    % Canny edge detection
    thresh = 0.1;
    edge_img = edge(img,'canny', thresh);
    
    % Note: The output from edge is an image of logical type.
    % Here we cast it to double before saving it.
    imwrite(im2double(edge_img), ['output/edge_' img_list{i} '.png']);
end


function runGenerateHoughAccumulator()
img_list = {'hough_1', 'hough_2', 'hough_3'};

% Value determined through trial and error.
theta_num_bins = 300;
for i = 1:length(img_list)
    img = imread(['output/edge_' img_list{i} '.png']);
    
    [y,x] = size(img);
    % Get the max value that the height can be. Formula from notes.
    max_height = ceil(sqrt(x*x+y*y));
    rho_num_bins = max_height*2;% *2 for + or - values
    
    hough_accumulator = generateHoughAccumulator(img,...
        theta_num_bins, rho_num_bins);
    
    % We'd like to save the hough accumulator array as an image to
    % visualize it. The values should be between 0 and 255 and the
    % data type should be uint8.
    imwrite(uint8(hough_accumulator), ['output/accumulator_' img_list{i} '.png']);
end


function runFindLines()
img_list = {'hough_1', 'hough_2', 'hough_3'};
%1: keep at 135
%2: keep at 80
%3: keep at 60
hough_threshold = [135,80,60];

for i = 1:length(img_list)
    orig_img = imread(['input/' img_list{i} '.png']);
    hough_img = imread(['output/accumulator_' img_list{i} '.png']);
    line_img = lineFinder(orig_img, hough_img, hough_threshold(i));
    
    % The values of line_img should be between 0 and 255 and the
    % data type should be uint8.
    %
    % Here we cast line_img to uint8 if you have not done so, otherwise
    % imwrite will treat line_img as a double image and save it to an
    % incorrect result.    
    imwrite(uint8(line_img), ['output/line_' img_list{i} '.png']);
end


function runFindLineSegments()

img_list = {'hough_1', 'hough_2', 'hough_3'};
hough_threshold = [135,80,60];

% img_list = {'hough_3'};
% hough_threshold = [135,80,60];

for i = 1:length(img_list)
    orig_img = imread(['input/' img_list{i} '.png']);
    hough_img = imread(['output/accumulator_' img_list{i} '.png']);
    edge_img = imread(['output/edge_' img_list{i} '.png']);
    line_img = lineSegmentFinder(orig_img, hough_img, hough_threshold(i), edge_img);
    
    % Note: The values of line_img should be between 0 and 255 and the
    % data type should be uint8.
    %
    % Here we cast line_img to uint8 if you have not done so, otherwise
    % imwrite will treat line_img as a double image and save it to an
    % incorrect result.        
    imwrite(uint8(line_img), ['output/croppedline_' img_list{i} '.png']);
end
