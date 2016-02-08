clear all
close all
clc

%I = imread('input/aloe.png');
%I = imread('input/cloth1.png');
%I = imread('input/lampshade.png');
%I = imread('input/mouse-pad.png');
%I = imread('input/arm-1.png');
%I = imread('input/arm-2.png');
%I = imread('input/baby2.png');
%I = imread('input/baby3.png');
%I = imread('input/banana-1.png');
%I = imread('input/blue-pan.png');
%I = imread('input/bowling1.png');
%I = imread('input/bowling2.png');
%I = imread('input/box-1.png');
%I = imread('input/box-2.png');
%I = imread('input/breakfast-1.png');
%I = imread('input/breakfast-2.png');
%I = imread('input/cloth2.png');
%I = imread('input/cloth3.png');
%I = imread('input/cloth4.png');
%I = imread('input/cones.png');
%I = imread('input/cups-1.png');
%I = imread('input/cups-2.png');
%I = imread('input/flowerpots.png');
%I = imread('input/garnics-1.png');
%I = imread('input/garnics-2.png');
%I = imread('input/garnics-3.png');
%I = imread('input/garnics-4.png');
%I = imread('input/ice-brick.png');
%I = imread('input/lampshade.png');
%I = imread('input/lampshade1.png');
%I = imread('input/lampshade2.png');
%I = imread('input/mid1.png');
%I = imread('input/mid2.png');
%I = imread('input/monopoly.png');
%I = imread('input/plastic.png');
%I = imread('input/red-plate.png');
%I = imread('input/rocks1.png');
%I = imread('input/rocks2.png');
%I = imread('input/sandwich-1.png');
%I = imread('input/sandwich-2.png');
%I = imread('input/scene01.png');
%I = imread('input/scene03.png');
%I = imread('input/scene04.png');
%I = imread('input/scene05.png');
%I = imread('input/scene09.png');
%I = imread('input/scene15.png');
%I = imread('input/table-1.png');
%I = imread('input/table-2.png');
%I = imread('input/teddy.png');
%I = imread('input/towel1.png');
%I = imread('input/tsukuba.png');
%I = imread('input/venus.png');
%I = imread('input/wood.png');
%I = imread('input/wood1.png');
I = imread('input/wood2.png');

II = rgb2gray(I);
J = entropyfilt(II);

%fname = 'texture.dat';
save 'texture.dat' J -ascii

imshow(II)
figure
imshow(J);

