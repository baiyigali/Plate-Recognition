%from pf
close all;clear all;clc;
tic;
image = imread('2.png');
gray = rgb2gray(image);
bw = edge(gray, 'sobel','horizontal');
theta = 0:179;
r = radon(bw,theta);
[m,n] = size(r);
c = 1 ;
for i = 1:m
    for j = 1:n        
        if r(1,1) < r(i,j)
            disp(i);
            disp(j);
            disp(r(i, j));
            r(1,1) = r(i,j);
            c = j;
        end
    end
end
rot = 90-c;
disp(rot);
subplot(3,1,1);imshow(image);
subplot(3,1,2);imshow(bw);
pic = imrotate(image,rot,'bilinear');
subplot(3,1,3);imshow(pic);
toc;