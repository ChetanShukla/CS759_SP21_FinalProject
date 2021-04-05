%Used to rename the initial image files. Only used once and not needed again.

output_num = 3;
for i = int16(3):int16(150)
    input_img_name = strcat('../images/image-', num2str(i), '.png');
    if isfile(input_img_name)
        movefile(input_img_name, strcat('../images/image-', num2str(output_num), '.png'));
        output_num = output_num + 1;
    end
end