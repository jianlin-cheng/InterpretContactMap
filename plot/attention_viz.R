library("scatterplot3d") # load
library(reshape2)
library(pheatmap)
L = 10
max_points_show = 3

a = cbind.data.frame('x'=1:L,'y'=1:L,'z'=1:L)
set.seed(10)
contact_map <- matrix(rnorm(L*L),L,L)
contact_map <- contact_map>1
pheatmap(contact_map+0,cluster_rows = F,cluster_cols = F)


b=expand.grid(a)
val = abs(rnorm(L*L*L))
for(i in 1:L){
  idx_x = a$x == i & a$z == i
  idx_y = a$y == i & a$z == i
  val[idx_x] <- val[idx_x] + abs(rnorm(sum(idx_x)))
  val[idx_y] <- val[idx_y] + abs(rnorm(sum(idx_y)))
}
b <- cbind(b,'val'=val)

for(i in 1:L){
  for(j in 1:L){
    z_idx <- b$x==i & b$y==j
    b$val[z_idx] <- exp(b$val[z_idx])/sum(exp(b$val[z_idx]))
  }
}
c <- b[as.logical(contact_map),]
c <- c[c$val>0.15,]
colors <- c()
for(i in 1:nrow(c)){
  colors <- c(colors,adjustcolor("black", alpha.f = 2*c$val[i]))
}


d <- melt(contact_map)
d <- d[d$value,]

e <- cbind.data.frame('x'=d$Var1,'y'=d$Var2,'z'=0,'col'='red','size'=2.5,'shape'=4)
f <- cbind.data.frame('x'=c$x,'y'=c$y,'z'=c$z,'col'=colors,'size'=10*c$val,'shape'=16)
plot_table <- rbind(e,f)
scatterplot3d(plot_table[,1:3], pch = plot_table$shape, color=plot_table$col,cex.symbols=plot_table$size)


library(rgl)
plot3d(plot_table$x,plot_table$y,plot_table$z,type='p')





