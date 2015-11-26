rm(list=ls())

library(doParallel)
library(foreach)
library(mixOmics)  
library(pROC)

# Original data
load(file='metaOtus.RData')
ID=meta$village

# Modelled data for publication
load(file='modelData.Rdata')

####
## Random Forests

library(randomForest)

cl=makeCluster(detectCores())
registerDoParallel(cl)
YModRF9=otusRF(otus,ID,nRep=200,nSeg=8,nInner=7,featRatio = 0.9)
stopCluster(cl)

# Extract average class predictions
yPred=YModRF9$yPred$maxModel[,,1]
# Extract class predictions per repetition
yPP=YModRF9$yPredPerRep$maxModel
yP1=yPP[,1,]
yP2=yPP[,2,]
yP3=yPP[,3,]
# Plot class probabilities
x=1:29
labels=meta$name
substr(labels,4,4)='-'
png(filename='class_probabilities_colblind.png',width=1024,height=768,pointsize=24)
par(mar=c(4.5,4,0,2)+.4)
matplot(x-.15,yP1,pch=20,col='#1b9e77',cex=0.5,xlab='',ylab='Class probability',main='',axes=F)
axis(1,labels=F,at=1:29,las=3)
text((1:29)-.55,-0.08,labels,pos=4,srt=-90,xpd=T)
axis(2,las=1)
axis(4,las=1)
box(bty='U')
matpoints(x,yP2,pch=20,col='#d95f02',cex=.5)
matpoints(x+.15,yP3,pch=20,col='#7570b3',cex=0.5)
points(x-.15,yPred[,1],pch=20,cex=1.5,col='#1b9e77')
points(x,yPred[,2],pch=20,cex=1.5,col='#d95f02')
points(x+.15,yPred[,3],pch=20,cex=1.5,col='#7570b3')
xWrong=c(6,22,23)+c(.15,-.15,-.15)
yWrong=c(yPred[6,3],yPred[22:23,1])
points(xWrong,yWrong,pch=1,cex=2.5,col='black')
dev.off()

### Permutation analysis
nPerm=400
missMax=numeric(nPerm)
cl=makeCluster(4)
registerDoParallel(cl)
for (perm in 1:nPerm) {
  cat(perm)
  cat('\n')
  # Randomly permute classification labels
  IDperm=sample(ID)
  # Model permuted data
  yPerm=otusRF(otus,IDperm,nRep=12,nSeg=7,nInner=6,featRatio = 0.75)
  # Extract number of misclassifications -> H0-population
  missMax[perm]=yPerm$misClass$maxModel
}
stopCluster(cl)

# parametric p-value (cumulative prob from Stud t-distributed H0)
pMiss=pt((YModRF9$misClass$maxModel-mean(missMax))/sd(missMax),length(missMax)-1)
# plot model vs H0
png(filename='Permutation.png',width=1024,height=768,pointsize=24)
par(mar=c(4,4,0,0+.4))
hist(missMax,10,freq=F,xlim=c(0,29),ylim=c(0,0.15),xlab='number of misclassifications (H0, n=400)',ylab='probability',main='')
lines(rep(YModRF9$misClass$maxModel,2),c(0,.09),lwd=2)
text(YModRF9$misClass$maxModel,.1,'Model',pos=3)
text(YModRF9$misClass$maxModel,.1,paste('p=',signif(pMiss,3),sep=''),pos=1)
text(mean(missMax),0.14,paste('mean(H0)=',signif(mean(missMax),3),sep=''),pos=3)
dev.off()

## Take out most informative features
VIPMin=YModRF9$VIPRank$minModel[order(YModRF9$VIPRank$minModel[,1]),]
VIPMid=YModRF9$VIPRank$midModel[order(YModRF9$VIPRank$midModel[,1]),]
VIPMax=YModRF9$VIPRank$maxModel[order(YModRF9$VIPRank$maxModel[,1]),]
write.csv2(VIPMax[1:16,],file='VIPMax.csv')

save(YModRF9,missMax,file='modelData.Rdata')
