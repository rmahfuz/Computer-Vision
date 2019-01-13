%run vlfeat-0.9.20/toolbox/vlsetup
img1 = double(rgb2gray(imread('pictures/pair4/1.jpg')));
img2 = double(rgb2gray(imread('pictures/pair4/2.jpg')));
%vl_version verbose
[f1,d1] = vl_sift(single(img1));
[f2,d2] = vl_sift(single(img2)) ;

%Writing to file:
fh = fopen('pictures/pair4/sift1.txt','w');
for i = 1:size(f1,2)
    fprintf(fh, '%d, %d, %d, %d, ', f1(1,i), f1(2,i), f1(3,i), f1(4,i));
    %fprintf(fh, '\n');
end
fclose(fh);

fh = fopen('pictures/pair4/sift2.txt','w');
for i = 1:size(f2,2)
    fprintf(fh, '%d, %d, %d, %d, ', f2(1,i), f2(2,i), f2(3,i), f2(4,i));
    %fprintf(fh, '\n');
end
fclose(fh);



