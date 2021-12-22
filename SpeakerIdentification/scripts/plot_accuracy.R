library(tidyverse)
library(ggplot2)
library(RColorBrewer)


data <- read_tsv("acc.txt")
ggplot(data,aes(fill=as.factor(split)))+
  geom_violin(aes(x=as.factor(split), y=acc,alpha=0.5))+
  geom_jitter(aes(x=as.factor(split), y=acc,color=as.factor(split)))+
  xlab("Test Split Ratio")+
  labs(fill='Test Split Ratio')+
  guides(color=FALSE,alpha=FALSE)+theme_bw()
  

