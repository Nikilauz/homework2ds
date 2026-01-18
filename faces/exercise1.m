%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                            %
% Practical assignment: Face detection                                       %
%                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                            %
%   Part 1:  load and prepare image the data                                 %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


imgpath='images';
respath='results';

colormap(gray)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 1: Loading and normalizing training images %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%% load training images
%%%%%%%%%%%%%%%%%
posfname=[imgpath '/possamples.mat'];
negfname=[imgpath '/negsamples.mat'];
load(posfname,'possamples'); npos=size(possamples,3);
load(negfname,'negsamples'); nneg=size(negsamples,3);  
possamples=double(possamples);
negsamples=double(negsamples);
fprintf('load  %6d positive samples from %s\n',npos,posfname)
fprintf('load  %6d negative samples from %s\n\n',nneg,negfname)


%%%%%%%%%%%%%%%% positive and negative training images are now 
%%%%%%%%%%%%%%%% inside 'possamples' and 'possamples' 3D arrays
%%%%%%%%%%%%%%%% of the size 24x24xN where
%%%%%%%%%%%%%%%%  - 24x24 is the pixel size of croped faces
%%%%%%%%%%%%%%%%  - N is the number of samples
%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%% It's always good to look at the data
%%%%%%%%%%%%%%%% let's display a few positive and negative samples
%%%%%%%%%%%%%%%%
possampleimage=[];
negsampleimage=[];
for i=1:10
  indpos=randperm(npos);
  indneg=randperm(nneg);
  posfaces=reshape(shiftdim(possamples(:,:,indpos(1:10)),1),24*10,24);
  negfaces=reshape(shiftdim(negsamples(:,:,indneg(1:10)),1),24*10,24);
  possampleimage=cat(2,possampleimage,posfaces);
  negsampleimage=cat(2,negsampleimage,negfaces);
end

clf
fprintf('Showing images of random Positive samples\n')
showimage(possampleimage'); title('positive samples')
fprintf('press a key...'), pause, fprintf('\n')
fprintf('Showing images of random Negative samples\n')
showimage(negsampleimage'); title('negative samples')
fprintf('press a key...'), pause, fprintf('\n')



%%%%%%%%%%%%%%%% we can also display an average face and non-face images
%%%%%%%%%%%%%%%%
clf, 
fprintf('Showing average images for Positive and Negative samples\n')
subplot(1,2,1), title('average positive image')
showimage(mean(possamples(:,:,1:2000),3)); 
subplot(1,2,2), title('average negative image')
showimage(mean(negsamples(:,:,1:2000),3)); 


%%%%%%%%%%%%%%%% Positive training images have been manually aligned
%%%%%%%%%%%%%%%% by x-y coordinates. However, they may have large
%%%%%%%%%%%%%%%% variation in the amplitude. To remove this variation
%%%%%%%%%%%%%%%% we normalize pixel values of all samples to zero mean
%%%%%%%%%%%%%%%% and variance value=1
%%%%%%%%%%%%%%%%
possamples=meanvarpatchnorm(possamples);
negsamples=meanvarpatchnorm(negsamples);


%%%%%%%%%%%%%%% SVM training and classification assumes each sample
%%%%%%%%%%%%%%% in the form of a vector. Let's flatten training images
%%%%%%%%%%%%%%% into one vector per sample
%%%%%%%%%%%%%%%
xsz=size(possamples,2);
ysz=size(possamples,1);
Xpos=transpose(reshape(possamples,ysz*xsz,npos));
Xneg=transpose(reshape(negsamples,ysz*xsz,nneg));


%%%%%%%%%%%%%%% make vectore with sample labels:
%%%%%%%%%%%%%%% +1 for positives
%%%%%%%%%%%%%%% -1 for negatives
%%%%%%%%%%%%%%%
ypos=ones(npos,1);
yneg=-ones(nneg,1);


%%%%%%%%%%%%%%% separate data into the training and validation set
%%%%%%%%%%%%%%% 
ntrainpos=1000;
ntrainneg=1000;
indpostrain=1:ntrainpos; indposval=indpostrain+ntrainpos;
indnegtrain=1:ntrainneg; indnegval=indnegtrain+ntrainneg;

Xtrain=[Xpos(indpostrain,:); Xneg(indnegtrain,:)];
ytrain=[ypos(indpostrain); yneg(indnegtrain)];
Xval=[Xpos(indposval,:); Xneg(indnegval,:)];
yval=[ypos(indposval); yneg(indnegval)];

% free memory
clear possamples negsamples



