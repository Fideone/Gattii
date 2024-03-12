# pan-genome analysis

```shell
#pan-genome analysis
conda install -c bioconda -y orthofinder
mkdir test && cd test
cp *.fasta  ./
cd ../
orthofinder -f /public3/home/sc73059/protein -t 128 -M msa   -a 128  -T raxml-ng
```

```R
#R plot
library(ggplot2)
library(RColorBrewer)
library(reshape2)
setwd('D:/ubuntu')
core <- read.table("curve_core.tsv")
pan <- read.table("curve_pan.tsv")
core <- melt(core,id.vars = "V1")
pan <- melt(pan,id.vars = "V1")
core <- core[,c(1,3)]
pan <- pan[,c(1,3)]
core[,3] <- "core genome"
pan[,3] <- "pan genome"
data <- rbind(core,pan)
#getPalette = colorRampPalette(brewer.pal(11, "PuOr"))
ggplot(data,aes(x=factor(V1),y=value,fill=V3)) +
  geom_boxplot(lwd=0.2,outlier.size = 0.3, width = 1.55, position =
                 position_dodge(width = 0.0)) +
 theme_bw() +
  theme(axis.text=element_text(color="black"),legend.title = element_blank(),panel.grid.major=element_blank(),panel.grid.minor=element_blank(),axis.text.x = element_text(angle = 45))
+   scale_x_discrete(breaks = c(0,10,20,30,40,50,60,70,80,90,100)) 
+ xlab("number of strains") + ylab("number of genes")
dev.off()


#heatmap

library("pheatmap")

library("RColorBrewer")


data<-read.table(file="泛基因组fig3a输入.csv“,header=TRUE,row.names= 1,sep=',')

head(data)


data <- as.matrix(data[,-1])  

head(data)


#heatmap

p <- pheatmap(data, scale = "none",cluster_rows = F,cluster_cols = F,
              color = colorRampPalette(c("black","lightblue","orange"))(100), gaps_row = c(43),
              gaps_col = c(4688,5555,7500),border_color=NA)




#PCA analysis

install.packages('usethis')
library(usethis)
install.packages('devtools')
library(devtools)
install_github("vqv/ggbiplot")
library(ggplot2)
library(plyr)
library(scales)
library(grid)
library(ggbiplot)
data <- read.csv("Fig.csv", header = TRUE, stringsAsFactors = FALSE)  
sample_names <- data[, 1]
num_columns <- data[, 2:32]
pca_data <- data.frame(sample_names)
pca_result <- prcomp(num_columns, scale. = TRUE)

ggbiplot(pca_result, 
         choices = c(1,2),
         obs.scale = 1, 
         var.scale = 1, 
         var.axes = FALSE, 
         groups = data2$Lineage, 
         ellipse = TRUE, 
         ellipse.prob = 0.95, 
         circle = F,pch = pca_data$sample_names) + 
  scale_color_manual(values = c('#8CB6EA','#F4B044','#66CCCC','#666699','#CCCC33','#FFCC99','#CC9999','#FF9900'))+
  theme_bw()


```

