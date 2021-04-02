%--------------------------------------------------------------------------
% Basic example
%--------------------------------------------------------------------------

%----------------------- 
% Edge detection
%-----------------------
% Image credit: CAVE Lab

img = imread('input/hello.png');
fh = figure;
subplot(2, 2, 1); imshow(img); title('Color Image');

% Convert the image to grayscale
gray_img = rgb2gray(img); 
subplot(2, 2, 2); imshow(gray_img); title('Grayscale Image');

% Convert the image to grayscale

% Sobel edge detection
thresh = 0.2;

edge_img = edge(gray_img,'sobel', thresh);
subplot(2, 2, 3); imshow(edge_img); title('Sobel Edge Detection');

% Canny edge detection
thresh = 0.5;

edge_img = edge(gray_img,'canny', thresh);
subplot(2, 2, 4); imshow(edge_img); title('Canny Edge Detection');
saveas(fh, 'output/hello_edges.png');

delete(fh);
