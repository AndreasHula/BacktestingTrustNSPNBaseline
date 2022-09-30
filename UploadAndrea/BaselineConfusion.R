library(nnet)
library(MASS)
library(party)
library(class)
library(ggplot2)
library(cdata)
library(dplyr)
library(plot.matrix)
library(pracma)
library(tiff)
library(png)
library(raster)
library('magick')
library(gridGraphics)
library(cowplot)
ChoiceTable=read.table("NSPNBaseInvTru.csv", sep =";", header=TRUE)
BaseChoice = ChoiceTable
BaseInvestor = matrix(0, length(BaseChoice[,1]),10)
BaseTrustee = matrix(0,length(BaseChoice[,1]),10)
for(j in 1:length(BaseChoice[,1]) ){
  i=j
  BaseInvestor[j,]=c(BaseChoice$Inv1[i],BaseChoice$Inv2[i], BaseChoice$Inv3[i], BaseChoice$Inv4[i],
                     BaseChoice$Inv5[i],BaseChoice$Inv6[i],BaseChoice$Inv7[i], BaseChoice$Inv8[i],
                     BaseChoice$Inv9[i],BaseChoice$Inv10[i])
  BaseTrustee[j,]=c(BaseChoice$Tru1[i],BaseChoice$Tru2[i], BaseChoice$Tru3[i],BaseChoice$Tru4[i],
                    BaseChoice$Tru5[i],BaseChoice$Tru6[i],BaseChoice$Tru7[i],BaseChoice$Tru8[i],
                    BaseChoice$Tru9[i],BaseChoice$Tru10[i])
  
}

Investor = matrix(0, length(BaseChoice[,1]),10)
Trustee = matrix(0, length(BaseChoice[,1]),10)
for(i in 1:length(BaseChoice[,1])){
  
  for( t in 1:10){ 
    Investor[i,t]=round(1/5*BaseInvestor[i,t])
  }
  for( t in 1:10){ 
    Trustee[i,t]=which.min( (1/6*3*5*Investor[i,t]*(0:4)-BaseTrustee[i,t])^2 )-1
  }
  
}




BaseInvestorI = readRDS(file="Investmentdata.rds")
SumInvestorI = rowSums(BaseInvestorI)

BaseTrusteeI = readRDS(file="Repaymentdata.rds")
SumTrusteeI = rowSums(BaseTrusteeI)
InvestorEarningsI = 200-SumInvestorI + SumTrusteeI


NSPNInvestorLikelihoodsI = readRDS(file="fullinvestor.rds")
NSPNInvestorToMI = readRDS(file="initialtom.rds")
BacktestInvestorToMI = readRDS(file="backtesttom.rds")
NSPNInvestorPlanI = readRDS(file="initialplan.rds")
BacktestInvestorPlanI = readRDS(file="backtestplan.rds")
NSPNInvestorAversionI = readRDS(file="initialaversion.rds")
BacktestInvestorAversionI = readRDS(file="backtestaversion.rds")
NSPNInvestorTempI = readRDS(file="initialtemp.rds")
NSPNInvestorTempI = 1/NSPNInvestorTempI
NSPNInvestorConfusTempI = readRDS(file="initialtemp.rds")
BacktestInvestorTempI = readRDS(file="backtesttemp.rds")
NSPNInvestorGuiltI = readRDS(file="initialguilt.rds")
BacktestInvestorGuiltI = readRDS(file="backtestguilt.rds")
NSPNInvestorIrritabilityI = readRDS(file="initialirr.rds")
BacktestInvestorIrritabilityI = readRDS(file="backtestirr.rds")
NSPNInvestorShiftI = readRDS(file="initialbelief.rds")
BacktestInvestorShiftI =readRDS(file="backtestbelief.rds")


RiskAversionConfusion = vector( "double" , 8*8)
k=0
Original_Risk_Aversion = {}
Recovered_Risk_Aversion = {}
for(i in 0:7){
  for(j in 0:7){
    k=k+1
    RiskAversionConfusion[k]=length(which(BacktestInvestorAversionI[which(NSPNInvestorAversionI==i)]==j) )/length(which(NSPNInvestorAversionI==i))
    Original_Risk_Aversion = c(Original_Risk_Aversion, 0.4+ 0.2*i)
    Recovered_Risk_Aversion = c(Recovered_Risk_Aversion, 0.4+ 0.2*j)
  }
}
C_Matrix = as.data.frame(RiskAversionConfusion)
confusion_matrix <- as.data.frame(table(BacktestInvestorAversionI, NSPNInvestorAversionI))
names(C_Matrix) = "Density"

ToMConfusion = vector( "double" , 3*3)
k=0
Original_ToM = {}
Recovered_ToM = {}
for(i in 0:2){
  for(j in 0:2){
    k=k+1
    ToMConfusion[k]=length(which(BacktestInvestorToMI[which(NSPNInvestorToMI==2*i)]==2*j) )/length(which(NSPNInvestorToMI==2*i))
    Original_ToM = c(Original_ToM, 2*i)
    Recovered_ToM = c(Recovered_ToM, 2*j)
  }
}
C_ToM_Matrix = as.data.frame(ToMConfusion)
ToM_confusion_matrix <- as.data.frame(table(BacktestInvestorToMI, NSPNInvestorToMI))
names(C_ToM_Matrix) = "Density"

PlanConfusion = vector( "double" , 4*4)
k=0
Original_Plan = {}
Recovered_Plan = {}
for(i in 1:4){
  for(j in 1:4){
    k=k+1
    PlanConfusion[k]=length(which(BacktestInvestorPlanI[which(NSPNInvestorPlanI==i)]==j) )/length(which(NSPNInvestorPlanI==i))
    Original_Plan = c(Original_Plan, i)
    Recovered_Plan = c(Recovered_Plan, j)
  }
}
C_Plan_Matrix = as.data.frame(PlanConfusion)
Plan_confusion_matrix <- as.data.frame(table(BacktestInvestorPlanI, NSPNInvestorPlanI))
names(C_Plan_Matrix) = "Density"

GuiltConfusion = vector( "double" , 3*3)
k=0
Original_Guilt = {}
Recovered_Guilt = {}
for(i in 0:2){
  for(j in 0:2){
    k=k+1
    GuiltConfusion[k]=length(which(BacktestInvestorGuiltI[which(NSPNInvestorGuiltI==i)]==j) )/length(which(NSPNInvestorGuiltI==i))
    Original_Guilt = c(Original_Guilt, i)
    Recovered_Guilt = c(Recovered_Guilt, j)
  }
}
C_Guilt_Matrix = as.data.frame(GuiltConfusion)
Guilt_confusion_matrix <- as.data.frame(table(BacktestInvestorGuiltI, NSPNInvestorGuiltI))
names(C_Guilt_Matrix) = "Density"

TempConfusion = vector( "double" , 4*4)
k=0
Original_Temp = {}
Recovered_Temp = {}
for(i in 1:4){
  for(j in 1:4){
    k=k+1
    TempConfusion[k]=length(which(BacktestInvestorTempI[which(NSPNInvestorConfusTempI==i)]==j) )/length(which(NSPNInvestorConfusTempI==i))
    Original_Temp = c(Original_Temp, i)
    Recovered_Temp = c(Recovered_Temp, j)
  }
}
C_Temp_Matrix = as.data.frame(TempConfusion)
Temp_confusion_matrix <- as.data.frame(table(BacktestInvestorTempI, NSPNInvestorConfusTempI))
names(C_Temp_Matrix) = "Density"

BacktestIrritationIndI = readRDS(file="Irrind.rds")

IrrConfusion = vector( "double" , 5*5)
k=0
Original_Irr = {}
Recovered_Irr = {}
for(i in 0:4){
  for(j in 0:4){
    k=k+1
    IrrConfusion[k]=length(which(BacktestInvestorIrritabilityI[which(NSPNInvestorIrritabilityI==i & BacktestIrritationIndI==1)]==j) )/length(which(NSPNInvestorIrritabilityI==i & BacktestIrritationIndI==1))
    Original_Irr = c(Original_Irr, i)
    Recovered_Irr = c(Recovered_Irr, j)
  }
}
C_Irr_Matrix = as.data.frame(IrrConfusion)
Irr_confusion_matrix <- as.data.frame(table(BacktestInvestorIrritabilityI[which(BacktestIrritationIndI==1)], NSPNInvestorIrritabilityI[which(BacktestIrritationIndI==1)]))
names(C_Irr_Matrix) = "Density"

ShiftConfusion = vector( "double" , 5*5)
k=0
Original_Shift = {}
Recovered_Shift = {}
for(i in 0:4){
  for(j in 0:4){
    k=k+1
    ShiftConfusion[k]=length(which(BacktestInvestorShiftI[which(NSPNInvestorShiftI==i)]==j) )/length(which(NSPNInvestorShiftI==i))
    Original_Shift = c(Original_Shift, i)
    Recovered_Shift = c(Recovered_Shift, j)
  }
}
C_Shift_Matrix = as.data.frame(ShiftConfusion)
Shift_confusion_matrix <- as.data.frame(table(BacktestInvestorShiftI, NSPNInvestorShiftI))
names(C_Shift_Matrix) = "Density"


prA=ggplot(data = C_Matrix,
          mapping = aes(x = Recovered_Risk_Aversion,
                        y = Original_Risk_Aversion))+
  geom_raster(aes(fill=Density)) +
  scale_x_continuous("Recovered Risk Aversion", 
                     breaks=c(0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8),
                     labels=c("0.4","0.6","0.8","1.0","1.2","1.4","1.6","1.8"))+
  scale_y_continuous("Original Risk Aversion",
                     breaks=c(0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8),
                     labels=c("0.4","0.6","0.8","1.0","1.2","1.4","1.6","1.8"))+
  labs( title="A) Risk Aversion Confusion Matrix ") +
  theme(axis.text.x=element_text(size=11, angle=0, vjust=0.3),
        axis.text.y=element_text(size=11),
        plot.title=element_text(size=17, hjust = 0.5))+
  theme(panel.border = element_blank())+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                               panel.background = element_blank(), axis.line = element_line(colour = "black"))

prT=ggplot(data = C_ToM_Matrix,
           mapping = aes(x = Recovered_ToM,
                         y = Original_ToM))+
  geom_raster(aes(fill=Density)) +
  scale_x_continuous("Recovered ToM", 
                     breaks=c(0,2,4),
                     labels=c("0","2","4"))+
  scale_y_continuous("Original ToM",
                     breaks=c(0,2,4),
                     labels=c("0","2","4"))+
  labs( title="B) ToM Confusion Matrix ") +
  theme(axis.text.x=element_text(size=11, angle=0, vjust=0.3),
        axis.text.y=element_text(size=11),
        plot.title=element_text(size=17, hjust = 0.5))+
  theme(panel.border = element_blank())+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                               panel.background = element_blank(), axis.line = element_line(colour = "black"))

prP=ggplot(data = C_Plan_Matrix,
           mapping = aes(x = Recovered_Plan,
                         y = Original_Plan))+
  geom_raster(aes(fill=Density)) +
  scale_x_continuous("Recovered Plan", 
                     breaks=c(1,2,3,4),
                     labels=c("1","2","3","4"))+
  scale_y_continuous("Original Plan",
                     breaks=c(1,2,3,4),
                     labels=c("1","2","3", "4"))+
  labs( title="C) Plan Confusion Matrix ") +
  theme(axis.text.x=element_text(size=11, angle=0, vjust=0.3),
        axis.text.y=element_text(size=11),
        plot.title=element_text(size=17, hjust = 0.5))+
  theme(panel.border = element_blank())+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                               panel.background = element_blank(), axis.line = element_line(colour = "black"))

prG=ggplot(data = C_Guilt_Matrix,
           mapping = aes(x = Recovered_Guilt,
                         y = Original_Guilt))+
  geom_raster(aes(fill=Density)) +
  scale_x_continuous("Recovered Guilt", 
                     breaks=c(0,1,2),
                     labels=c("0","0.4","1"))+
  scale_y_continuous("Original Guilt",
                     breaks=c(0,1,2),
                     labels=c("0","0.4","1"))+
  labs( title="D) Guilt Confusion Matrix ") +
  theme(axis.text.x=element_text(size=11, angle=0, vjust=0.3),
        axis.text.y=element_text(size=11),
        plot.title=element_text(size=17, hjust = 0.5))+
  theme(panel.border = element_blank())+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                               panel.background = element_blank(), axis.line = element_line(colour = "black"))

prTe=ggplot(data = C_Temp_Matrix,
           mapping = aes(x = Recovered_Temp,
                         y = Original_Temp))+
  geom_raster(aes(fill=Density)) +
  scale_x_continuous("Recovered Temp", 
                     breaks=c(1,2,3,4),
                     labels=c("1","2","3","4"))+
  scale_y_continuous("Original Temp",
                     breaks=c(1,2,3,4),
                     labels=c("1","2","3", "4"))+
  labs( title="E) Temperature Confusion Matrix ") +
  theme(axis.text.x=element_text(size=11, angle=0, vjust=0.3),
        axis.text.y=element_text(size=11),
        plot.title=element_text(size=17, hjust = 0.5))+
  theme(panel.border = element_blank())+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                               panel.background = element_blank(), axis.line = element_line(colour = "black"))

prI=ggplot(data = C_Irr_Matrix,
           mapping = aes(x = Recovered_Irr,
                         y = Original_Irr))+
  geom_raster(aes(fill=Density)) +
  scale_x_continuous("Recovered Irritability", 
                     breaks=c(0, 1,2,3,4),
                     labels=c("0","1/4","2/4","3/4","1"))+
  scale_y_continuous("Original Irritability",
                     breaks=c(0,1,2,3,4),
                     labels=c("0", "1/4","2/4","3/4", "1"))+
  labs( title="F) Irritability Confusion Matrix ") +
  theme(axis.text.x=element_text(size=11, angle=0, vjust=0.3),
        axis.text.y=element_text(size=11),
        plot.title=element_text(size=17, hjust = 0.5))+
  theme(panel.border = element_blank())+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                               panel.background = element_blank(), axis.line = element_line(colour = "black"))

prS=ggplot(data = C_Shift_Matrix,
           mapping = aes(x = Recovered_Shift,
                         y = Original_Shift))+
  geom_raster(aes(fill=Density)) +
  scale_x_continuous("Recovered Beliefs", 
                     breaks=c(0, 1,2,3,4),
                     labels=c("0","1","2","3","4"))+
  scale_y_continuous("Original Beliefs",
                     breaks=c(0,1,2,3,4),
                     labels=c("0", "1","2","3", "4"))+
  labs( title="G) Irritation Belief Confusion Matrix ") +
  theme(axis.text.x=element_text(size=11, angle=0, vjust=0.3),
        axis.text.y=element_text(size=11),
        plot.title=element_text(size=17, hjust = 0.5))+
  theme(panel.border = element_blank())+ theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                               panel.background = element_blank(), axis.line = element_line(colour = "black"))


#png(file="ConfRev.png", width = 40, height=40 , units="in" , res=300)
#par(mfrow=c(3,3))
plot_grid(prA, prT, prP, prG, prTe, prI, prS)
#dev.off()