BIC_full = sum(NSPNInvestorLikelihoodsI+7/2*(log(10)-log(2*pi)))

Riskonly = readRDS(file="riskonlyinvestor.rds")

BIC_risk = sum(Riskonly + 5/2*(log(10)-log(2*pi)))

ToMonly = readRDS(file="tomonlyinvestor.rds")

BIC_ToM = sum(ToMonly+3/2*(log(10)-log(2*pi)) )

Zeroonly = readRDS( file="zeroonlyinvestor.rds")

BIC_Zero = sum(Zeroonly+2/2*(log(10)-log(2*pi)) )

Random_Baseline = -log(1/5)*10*784