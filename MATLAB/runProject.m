function runProject(varargin)
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

    fun_handles = {@runPreprocess,@runHoughAccumulator,@runProcessGPUHough};
    % Call test harness
    runTests(varargin, fun_handles);
end

function runPreprocess()
    preprocess
end

function runHoughAccumulator()

    hough_output_folder_name = '../processed_images/hough/MATLAB';
    % Create the output folder if it doesn't exist
    if ~exist(hough_output_folder_name, 'dir')
       mkdir(hough_output_folder_name)
    end
    
    bloodcell_output_folder_name = '../processed_images/bloodcells/MATLAB';
    % Create the output folder if it doesn't exist
    if ~exist(bloodcell_output_folder_name, 'dir')
       mkdir(bloodcell_output_folder_name)
    end
    
    input_folder_name = '../processed_images/edges/MATLAB';
    original_folder_name = '../images';

    shrink_factor = 4;
    radius_start = 20/shrink_factor;
    for i = int16(1):int16(100)

        % Read the image.
        img = imread(strcat(input_folder_name,'/image-', num2str(i), '.png'));

        acc = houghAccumulator(img,shrink_factor,radius_start);

        % To write the accumulator as an image, scale all values to be
        % between 0 and 255.
        imwrite(uint8(rescale(acc,0,255)), strcat(hough_output_folder_name,'/image-',num2str(i),'.png'));
        
        circles = findCircles(acc);
        circles(:,1:2) = circles(:,1:2)*shrink_factor-1;
        circles(:,3) = (circles(:,3)+radius_start)*shrink_factor;
        
        img_circles = drawCircles(imread(strcat(original_folder_name,'/image-', num2str(i), '.png')),circles); 
        imwrite(uint8(img_circles),strcat(bloodcell_output_folder_name,'/image-', num2str(i), '.png'));

    end
end

function runProcessGPUHough()
    hough_output_folder_name = '../processed_images/hough';
    % Create the output folder if it doesn't exist
    if ~exist(hough_output_folder_name, 'dir')
       mkdir(hough_output_folder_name)
    end
    
    bloodcell_output_folder_name = '../processed_images/bloodcells';
    % Create the output folder if it doesn't exist
    if ~exist(bloodcell_output_folder_name, 'dir')
       mkdir(bloodcell_output_folder_name)
    end
    
    input_folder_name = '../processed_images/hough/binary';
    original_folder_name = '../images';

    shrink_factor = 4;
    radius_start = 20/shrink_factor;
    for i = int16(1):int16(100)
        file = fopen(strcat(input_folder_name,'/image-', num2str(i), '-out'));
        cuda_acc_1 = fread(file,[64,64], 'int32');
        cuda_acc_2 = fread(file,[64,64], 'int32');
        cuda_acc_3 = fread(file,[64,64], 'int32');
        fclose(file);

        acc = cat(3,cuda_acc_1,cuda_acc_2,cuda_acc_3);

        % To write the accumulator as an image, scale all values to be
        % between 0 and 255.
        imwrite(uint8(rescale(acc,0,255)), strcat(hough_output_folder_name,'/image-',num2str(i),'.png'));
        
        circles = findCircles(acc);
        circles(:,1:2) = circles(:,1:2)*shrink_factor-1;
        circles(:,3) = (circles(:,3)+radius_start)*shrink_factor;
        
        img_circles = drawCircles(imread(strcat(original_folder_name,'/image-', num2str(i), '.png')),circles); 
        imwrite(uint8(img_circles),strcat(bloodcell_output_folder_name,'/image-', num2str(i), '.png'));

    end
end

