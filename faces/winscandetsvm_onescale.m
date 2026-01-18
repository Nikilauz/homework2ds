function [bbox,conf,normpatches]=winscandetsvm_onescale(img,model,wxsz,wysz,confthresh);

%
% [bbox,conf]=winscandetsvm_onescale(img,model,confthresh);
%

if nargin<5 confthresh=-inf; end

[ysz,xsz]=size(img);
[x,y]=meshgrid(1:xsz-wxsz,1:ysz-wysz);
bbox=[x(:) y(:) x(:)+wxsz-1 y(:)+wysz-1];
n=size(bbox,1);

fprintf('evaluating %d subwindows\n',n)

imgcropall=zeros(wysz,wxsz,length(x(:)));
for i=1:size(bbox,1)
  imgcropall(:,:,i)=cropbbox(img,bbox(i,:));
end
imgcropall=meanvarpatchnorm(imgcropall);
X=[reshape(imgcropall,wysz*wxsz,n)];

% run classification
conf=X'*model.W-model.b;

ind=find(conf>confthresh);
bbox=bbox(ind,:);
conf=conf(ind);
normpatches=imgcropall(:,:,ind);