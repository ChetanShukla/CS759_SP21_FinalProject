function img_circles = drawCircles(img,circles)
    % Create figure and show the image so that we can draw lines on top of it.
    fh = figure();
    figure(fh);
    imshow(img);
    hold on;
    th = 0:pi/50:2*pi;
    for i = 1:size(circles)
        x = circles(i,1);
        y = circles(i,2);
        r = circles(i,3);
        xunit = r * cos(th) + x;
        yunit = r * sin(th) + y;
        plot(xunit, yunit,'r-','LineWidth',3);
    end
    hold off;
    % Save the image with the lines drawn on it.
    img_circles = saveAnnotatedImg(fh);
    delete(fh);

    % fh = figure();
    % subplot(2,2,1)
    % imshow(imread(strcat('../images', im_name)));
    % subplot(2,2,2)
    % imshow(imread(strcat('../processed_images/edges',im_name)));
    % % subplot(2,2,3)
    % % imshow(im2double(uint8(acc)));
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
