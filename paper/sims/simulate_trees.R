library(TreeSim)
library(phyclust)
library(readr)
library(ggtree)
library(phytools)
library(paleotree)

n <- 1000
lambdasky <- c(8,4)
deathsky <- c(4,4)
sampprobsky <- c(0.25,0.25)
timesky <- c(0,1)

options <- "-mHKY -l1000 -on"
simulate_tree <- function(i){
  root_height <- 0
  while (root_height < 1.2){
    tree <- sim.bdsky.stt(n,lambdasky,deathsky,timesky, sampprobsky)[[1]]
    heights <- dateNodes(tree)
    root_height <- max(heights)
    print(root_height)
  }
  dates <- root_height - heights[1:n]
  
  for (j in 1:n){
    tree$tip.label[j] = paste(tree$tip.label[j], toString(dates[j]), sep="_")
  }
  write.tree(tree, file=paste("trees/sim",i,".tree", sep=""))
  
  seq <- seqgen(opts=options, rooted.tree=tree)
  write_lines(seq, paste0("seqs/sim",i,".nexus"))
  return(tree)
}

trees <- lapply(1:1,simulate_tree)
ggtree(trees[[1]]) + theme_tree2() +geom_tiplab()

get_height <- function(tree){
  return(max(nodeHeights(tree)))
}