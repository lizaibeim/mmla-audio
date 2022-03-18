library(tidyverse)
library(ggplot2)
library(RColorBrewer)


data <- read_tsv("acc.txt")
ggplot(data,aes(fill=as.factor(segment)))+
  geom_violin(aes(x=as.factor(segment), y=acc,alpha=0.5))+
  geom_jitter(aes(x=as.factor(segment), y=acc,color=as.factor(segment)))+
  xlab("Test Split Ratio")+
  labs(fill='Test Split Ratio')+
  guides(color=FALSE,alpha=FALSE)+theme_bw()
  

