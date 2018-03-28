close all;clear all;clc;
tic;
img = imread('s4.png');
%img = imread('D:\Dataset\PVCTest\car_forehead\fh7.png');

gray = rgb2gray(img);
bw = edge(gray, 'sobel',0.01, 'vertical');
[H,Theta,Rho] = hough(bw);
subplot(222), imshow(H,[],'XData',Theta,'YData',Rho,'InitialMagnification','fit'),...
    title('rho\_theta space and peaks');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
% label the top 5 intersections
P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
x = Theta(P(:,2)); 
y = Rho(P(:,1));
plot(x,y,'*','color','r');

lines = houghlines(bw,Theta,Rho,P,'FillGap',5,'MinLength',7);
figure, imshow(img), hold on
max_len = 0;
for k = 1:length(lines)
 xy = [lines(k).point1; lines(k).point2];
 plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
end
toc;