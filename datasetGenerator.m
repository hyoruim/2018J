close all;
clear all;
clc;

numData = 5000;

%% circles

circles = zeros(numData,2501);
for i=1:numData
    
    ran1=randperm(48,1);
    ran2=randperm(48,1);
    ran3=7+randperm(5,1);
    randSigma=randperm(5,1)/10;
    
    I=ones(50,50);
    circle = insertShape(I, 'FilledCircle', [ran1 ran2 ran3], 'Color','black');
    circle = circle+normrnd(0, randSigma, size(I)); % AWGN noise
    circle=rgb2gray(circle);
    
    imwrite(circle,['circle_' num2str(i) '.bmp']);
    %imshow(circle);
    
    circleRow = reshape(circle,1,2500);
    circles(i,1:2501) = [0 circleRow];
    
end

    csvwrite('circle.csv', circles);
    

%% rectangles

rects = zeros(numData,2501);
for i=1:numData
    
    ran1=randperm(48,1);
    ran2=randperm(48,1);
    ran3=13+randperm(15,1);
    ran4=13+randperm(15,1);
    randSigma=randperm(5,1)/10;
    
    I=ones(50,50);
    rect = insertShape(I, 'FilledRectangle', [ran1 ran2 ran3 ran4], 'Color','black');
    rect = rect + normrnd(0, randSigma, size(I)); % AWGN noise
    rect=rgb2gray(rect);
    
    imwrite(rect,['rect_' num2str(i) '.bmp']);
    %imshow(rect);
    
    rectRow = reshape(rect,1,2500);
    rects(i,1:2501) = [1 rectRow];
    
end

 csvwrite('rect.csv',rects);
 
 csvwrite('test_set.csv',[rects;circles]);