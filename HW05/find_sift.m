%run vlfeat-0.9.20/toolbox/vlsetup
img = double(rgb2gray(imread('pictures/5.jpg')));
%vl_version verbose
%[f1,d1] = vl_sift(single(img), 'edgethresh', 2);
[f1,d1] = vl_sift(single(img));

%Writing to file:
fh = fopen('sift5.txt','w');
for i = 1:size(f1,2)
    fprintf(fh, '%d, %d, %d, %d, ', f1(1,i), f1(2,i), f1(3,i), f1(4,i));
    %fprintf(fh, '\n');
end
fclose(fh);