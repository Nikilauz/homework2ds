function res=meanvarpatchnorm(images)
  
[ysz,xsz,nsz]=size(images);

imgs=reshape(images,xsz*ysz,nsz);
meanval=mean(imgs,1);
stdval=std(imgs,0,1);
imgs=imgs-repmat(meanval,xsz*ysz,1);
imgs=imgs./(eps+repmat(stdval,xsz*ysz,1));
res=reshape(imgs,ysz,xsz,nsz);
