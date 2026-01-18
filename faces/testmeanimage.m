%%%%%%%%%%%%%%% W of the best classifier "looks like" and average face image
%%%%%%%%%%%%%%% Is it sufficient to construct W as the mean of of positive
%%%%%%%%%%%%%%% samples??? Let's try it
%%%%%%%%%%%%%%%


Walt=transpose(mean(Xtrain(ytrain==1,:),1));
clf, showimage(reshape(Walt,24,24)), drawnow

balt=0;
conftrainnew = Xtrain*Walt-balt;
confvalnew   = Xval*Walt-balt;
acctrainnew = mean((conftrainnew>0)*2-1==ytrain);
accvalnew   = mean((confvalnew>0)*2-1==yval);
fprintf('Training and validation accuracy estimated from W = "average image": %1.3f; %1.3f\n',acctrainnew,accvalnew)
fprintf('\n')





