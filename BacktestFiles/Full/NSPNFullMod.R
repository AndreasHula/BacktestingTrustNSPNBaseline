#BaseChoiceTable = read.csv(file="BaseFullTrust.csv", sep=";", header=TRUE)

#BaseChoiceTable= BaseChoiceTable[]

#Do= data.frame(  cbind(BaseChoiceTable$ID, FullInvestorLikelihoods,
#                      FullInvestorToM, FullInvestorPlan, 
#                      FullInvestorGuilt*(0.1*FullInvestorGuilt +0.3), 
#                      0.4+0.2*FullInvestorAversion, 1/FullInvestorTemp , 
#                      FullInvestorShift, 1/4*FullInvestorIrritability, IrritationIndR) )
#names(D) = c("ID", "NLL", "InvestorToM", "InvestorPlan", "InvestorGuilt", 
#             "InvestorAversion", "InvestorTemp", "InvestorIrrBelief", 
#             "InvestorIrritability", "IrrInd")

X = (0);


n = 824;
m = 824;
FullInvestorLikelihoods=vector("double",m)
FullInvestorToM=vector("double",m)
FullInvestorGuilt=vector("double",m)
FullInvestorAversion=vector("double",m)       
FullInvestorPlan=vector("double",m)
FullInvestorTemp=vector("double",m)
FullInvestorShift=vector("double",m)
FullInvestorIrritability=vector("double",m)
FullTrusteeLikelihoods=vector("double",m)
FullTrusteeToM=vector("double",m)
FullTrusteeGuilt=vector("double",m)
FullTrusteeAversion=vector("double",m)
FullTrusteePlan=vector("double",m)
FullTrusteeTemp=vector("double",m)
FullTrusteeShift=vector("double",m)
FullTrusteeIrritability=vector("double",m)
FullIExp = array(0, c(m,10, 5))
FullIBelief = array(0, c(5,3,m,10))
FullIIrr = array(0, c(5,5,m,10))
FullIShift = array(0, c(6,5,m,10))
FullIAct = array(0, c(5,10,3,5,5,m))
FullTExp = array(0, c(m,10, 5))
FullTBelief = array(0, c(5,3,m,10))
FullTIrr = array(0, c(5,5,m,10))
FullTShift = array(0, c(6,5,m,10))
FullTAct = array(0, c(5,10,3,5,5,m))
for( j in 1:(length(X))){
fid =file(paste0('DeGNSPNFullMod', toString(X[j]), '.bin'),'rb');
l=n

for( i in 1:l){
    FullInvestorLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullInvestorToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullInvestorGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullInvestorAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");      
    FullInvestorPlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullInvestorTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullInvestorShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullInvestorIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullTrusteeLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullTrusteeToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullTrusteeGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullTrusteeAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullTrusteePlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullTrusteeTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullTrusteeShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullTrusteeIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    for( t in 1:10){
        for( k in 1:5){
            FullIExp[i+(j-1)*n,t,k] =  readBin(fid,double(),endian = "little");
            for( g in 1:3){
                FullIBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");        
            }
            for( irr in 1:5){
                FullIIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                FullIShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                 FullIShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
            }
            for( g in 1:3){     
                 for( irr in 1:5){
                     for( act in 1:5){
                         FullIAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
                     }
                 }
            }
        }
    }
    for( t in 1:10){
        for( k in 1:5){        
            FullTExp[i+(j-1)*n,t,k] = readBin(fid,double(),endian = "little");
            for( g in 1:3){
                FullTBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");         
            }
            for( irr in 1:5){
                FullTIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                FullTShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                FullTShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
            }
            for( g in 1:3){     
                 for( irr in 1:5){
                     for( act in 1:5){
                         FullTAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
                     }
                 }
            }            
        }
    }
}
    close(fid);
}
X = (0);


n = 824;
m = 824;
FullZInvestorLikelihoods=vector("double",m)
FullZInvestorToM=vector("double",m)
FullZInvestorGuilt=vector("double",m)
FullZInvestorAversion=vector("double",m)       
FullZInvestorPlan=vector("double",m)
FullZInvestorTemp=vector("double",m)
FullZInvestorShift=vector("double",m)
FullZInvestorIrritability=vector("double",m)
FullZTrusteeLikelihoods=vector("double",m)
FullZTrusteeToM=vector("double",m)
FullZTrusteeGuilt=vector("double",m)
FullZTrusteeAversion=vector("double",m)
FullZTrusteePlan=vector("double",m)
FullZTrusteeTemp=vector("double",m)
FullZTrusteeShift=vector("double",m)
FullZTrusteeIrritability=vector("double",m)
FullZIExp = array(0, c(m,10, 5))
FullZIBelief = array(0, c(5,3,m,10))
FullZIIrr = array(0, c(5,5,m,10))
FullZIShift = array(0, c(6,5,m,10))
FullZIAct = array(0, c(5,10,3,5,5,m))
FullZTExp = array(0, c(m,10, 5))
FullZTBelief = array(0, c(5,3,m,10))
FullZTIrr = array(0, c(5,5,m,10))
FullZTShift = array(0, c(6,5,m,10))
FullZTAct = array(0, c(5,10,3,5,5,m))
for( j in 1:(length(X))){
  fid =file(paste0('DeGNSPNFullModNull', toString(X[j]), '.bin'),'rb');
  l=n
  
  for( i in 1:l){
    FullZInvestorLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullZInvestorToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZInvestorGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZInvestorAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");      
    FullZInvestorPlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZInvestorTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullZInvestorShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZInvestorIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZTrusteeLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullZTrusteeToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZTrusteeGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZTrusteeAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZTrusteePlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZTrusteeTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    FullZTrusteeShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    FullZTrusteeIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    for( t in 1:10){
      for( k in 1:5){
        FullZIExp[i+(j-1)*n,t,k] =  readBin(fid,double(),endian = "little");
        for( g in 1:3){
          FullZIBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");        
        }
        for( irr in 1:5){
          FullZIIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
          FullZIShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
          FullZIShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
        }
        for( g in 1:3){     
          for( irr in 1:5){
            for( act in 1:5){
              FullZIAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
            }
          }
        }
      }
    }
    for( t in 1:10){
      for( k in 1:5){        
        FullZTExp[i+(j-1)*n,t,k] = readBin(fid,double(),endian = "little");
        for( g in 1:3){
          FullZTBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");         
        }
        for( irr in 1:5){
          FullZTIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
          FullZTShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
          FullZTShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
        }
        for( g in 1:3){     
          for( irr in 1:5){
            for( act in 1:5){
              FullZTAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
            }
          }
        }            
      }
    }
  }
  close(fid);
}

FullFInvestorLikelihoods = vector("double",824)
FullFTrusteeLikelihoods = vector("double", 824)
for(i in 1:824){
  FullFInvestorLikelihoods[i] = min(FullInvestorLikelihoods[i], FullZInvestorLikelihoods[i] )
  FullFTrusteeLikelihoods[i] = min(FullTrusteeLikelihoods[i], FullZTrusteeLikelihoods[i] )
}
FullZInvestorLikelihoodsI = FullZInvestorLikelihoods[which(IndicatorDemog==1)]
FullZTrusteeLikelihoodsI = FullZTrusteeLikelihoods[which(IndicatorDemog==1)]
FullFInvestorLikelihoodsI = FullFInvestorLikelihoods[which(IndicatorDemog==1)]
FullFTrusteeLikelihoodsI = FullFTrusteeLikelihoods[which(IndicatorDemog==1)]