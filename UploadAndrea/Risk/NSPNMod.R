#BaseChoiceTable = read.csv(file="BaseRiskTrust.csv", sep=";", header=TRUE)

#BaseChoiceTable= BaseChoiceTable[]

#Do= data.frame(  cbind(BaseChoiceTable$ID, RiskInvestorLikelihoods,
#                      RiskInvestorToM, RiskInvestorPlan, 
#                      RiskInvestorGuilt*(0.1*RiskInvestorGuilt +0.3), 
#                      0.4+0.2*RiskInvestorAversion, 1/RiskInvestorTemp , 
#                      RiskInvestorShift, 1/4*RiskInvestorIrritability, IrritationIndR) )
#names(D) = c("ID", "NLL", "InvestorToM", "InvestorPlan", "InvestorGuilt", 
#             "InvestorAversion", "InvestorTemp", "InvestorIrrBelief", 
#             "InvestorIrritability", "IrrInd")

X = (0:7);


n = 103;
m = 824;
RiskInvestorLikelihoods=vector("double",m)
RiskInvestorToM=vector("double",m)
RiskInvestorGuilt=vector("double",m)
RiskInvestorAversion=vector("double",m)       
RiskInvestorPlan=vector("double",m)
RiskInvestorTemp=vector("double",m)
RiskInvestorShift=vector("double",m)
RiskInvestorIrritability=vector("double",m)
RiskTrusteeLikelihoods=vector("double",m)
RiskTrusteeToM=vector("double",m)
RiskTrusteeGuilt=vector("double",m)
RiskTrusteeAversion=vector("double",m)
RiskTrusteePlan=vector("double",m)
RiskTrusteeTemp=vector("double",m)
RiskTrusteeShift=vector("double",m)
RiskTrusteeIrritability=vector("double",m)
RiskIExp = array(0, c(m,10, 5))
RiskIBelief = array(0, c(5,3,m,10))
RiskIIrr = array(0, c(5,5,m,10))
RiskIShift = array(0, c(6,5,m,10))
RiskIAct = array(0, c(5,10,3,5,5,m))
RiskTExp = array(0, c(m,10, 5))
RiskTBelief = array(0, c(5,3,m,10))
RiskTIrr = array(0, c(5,5,m,10))
RiskTShift = array(0, c(6,5,m,10))
RiskTAct = array(0, c(5,10,3,5,5,m))
for( j in 1:(length(X))){
fid =file(paste0('DeGNSPNMod', toString(X[j]), '.bin'),'rb');
l=n

for( i in 1:l){
    RiskInvestorLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    RiskInvestorToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskInvestorGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskInvestorAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");      
    RiskInvestorPlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskInvestorTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    RiskInvestorShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskInvestorIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskTrusteeLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    RiskTrusteeToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskTrusteeGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskTrusteeAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskTrusteePlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskTrusteeTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
    RiskTrusteeShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    RiskTrusteeIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
    for( t in 1:10){
        for( k in 1:5){
            RiskIExp[i+(j-1)*n,t,k] =  readBin(fid,double(),endian = "little");
            for( g in 1:3){
                RiskIBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");        
            }
            for( irr in 1:5){
                RiskIIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                RiskIShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                 RiskIShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
            }
            for( g in 1:3){     
                 for( irr in 1:5){
                     for( act in 1:5){
                         RiskIAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
                     }
                 }
            }
        }
    }
    for( t in 1:10){
        for( k in 1:5){        
            RiskTExp[i+(j-1)*n,t,k] = readBin(fid,double(),endian = "little");
            for( g in 1:3){
                RiskTBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");         
            }
            for( irr in 1:5){
                RiskTIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                RiskTShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                RiskTShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
            }
            for( g in 1:3){     
                 for( irr in 1:5){
                     for( act in 1:5){
                         RiskTAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
                     }
                 }
            }            
        }
    }
}
    close(fid);
}


X = (0:1);


n = 412;
m = 824;
RiskZInvestorLikelihoods=vector("double",m)
RiskZInvestorToM=vector("double",m)
RiskZInvestorGuilt=vector("double",m)
RiskZInvestorAversion=vector("double",m)       
RiskZInvestorPlan=vector("double",m)
RiskZInvestorTemp=vector("double",m)
RiskZInvestorShift=vector("double",m)
RiskZInvestorIrritability=vector("double",m)
RiskZTrusteeLikelihoods=vector("double",m)
RiskZTrusteeToM=vector("double",m)
RiskZTrusteeGuilt=vector("double",m)
RiskZTrusteeAversion=vector("double",m)
RiskZTrusteePlan=vector("double",m)
RiskZTrusteeTemp=vector("double",m)
RiskZTrusteeShift=vector("double",m)
RiskZTrusteeIrritability=vector("double",m)
RiskZIExp = array(0, c(m,10, 5))
RiskZIBelief = array(0, c(5,3,m,10))
RiskZIIrr = array(0, c(5,5,m,10))
RiskZIShift = array(0, c(6,5,m,10))
RiskZIAct = array(0, c(5,10,3,5,5,m))
RiskZTExp = array(0, c(m,10, 5))
RiskZTBelief = array(0, c(5,3,m,10))
RiskZTIrr = array(0, c(5,5,m,10))
RiskZTShift = array(0, c(6,5,m,10))
RiskZTAct = array(0, c(5,10,3,5,5,m))
for( j in 1:(length(X))){
    fid =file(paste0('DeGNSPNModNull', toString(X[j]), '.bin'),'rb');
    l=n
    
    for( i in 1:l){
        RiskZInvestorLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
        RiskZInvestorToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZInvestorGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZInvestorAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");      
        RiskZInvestorPlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZInvestorTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
        RiskZInvestorShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZInvestorIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZTrusteeLikelihoods[i+(j-1)*n]=readBin(fid,double(),endian = "little");
        RiskZTrusteeToM[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZTrusteeGuilt[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZTrusteeAversion[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZTrusteePlan[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZTrusteeTemp[i+(j-1)*n]=readBin(fid,double(),endian = "little");
        RiskZTrusteeShift[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        RiskZTrusteeIrritability[i+(j-1)*n]=readBin(fid,integer(),endian = "little");
        for( t in 1:10){
            for( k in 1:5){
                RiskZIExp[i+(j-1)*n,t,k] =  readBin(fid,double(),endian = "little");
                for( g in 1:3){
                    RiskZIBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");        
                }
                for( irr in 1:5){
                    RiskZIIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                    RiskZIShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                    RiskZIShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                }
                for( g in 1:3){     
                    for( irr in 1:5){
                        for( act in 1:5){
                            RiskZIAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
                        }
                    }
                }
            }
        }
        for( t in 1:10){
            for( k in 1:5){        
                RiskZTExp[i+(j-1)*n,t,k] = readBin(fid,double(),endian = "little");
                for( g in 1:3){
                    RiskZTBelief[k,g,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");         
                }
                for( irr in 1:5){
                    RiskZTIrr[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                    RiskZTShift[k,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                    RiskZTShift[k+1,irr,i+(j-1)*n,t]=  readBin(fid,double(),endian = "little");
                }
                for( g in 1:3){     
                    for( irr in 1:5){
                        for( act in 1:5){
                            RiskZTAct[act,t,g,k,irr,i+(j-1)*n] = readBin(fid,double(),endian = "little");
                        }
                    }
                }            
            }
        }
    }
    close(fid);
}

RiskFInvestorLikelihoods = vector("double",824)
RiskFTrusteeLikelihoods = vector("double", 824)
for(i in 1:824){
    RiskFInvestorLikelihoods[i] = min(RiskInvestorLikelihoods[i], RiskZInvestorLikelihoods[i] )
    RiskFTrusteeLikelihoods[i] = min(RiskTrusteeLikelihoods[i], RiskZTrusteeLikelihoods[i] )
}
RiskFInvestorLikelihoodsI = RiskFInvestorLikelihoods[which(IndicatorDemog==1)]
RiskFTrusteeLikelihoodsI = RiskFTrusteeLikelihoods[which(IndicatorDemog==1)]