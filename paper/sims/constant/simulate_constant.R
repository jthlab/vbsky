library(TreeSim)
library(phyclust)
library(readr)
library(ggtree)
library(phytools)
library(paleotree)

n <- 0
timestop <- 4
lambdasky <- c(5.2,5.2)
deathsky <- c(4,4)
sampprobsky <- c(0.25,0.25)
timesky <- c(0,1)

options <- "-mHKY -l1000 -on"
simulate_tree <- function(i){
  tree <- 0
  while (is.numeric(tree)) {
    root_height <- 0
    tree <- sim.bdsky.stt(0,lambdasky,deathsky,timesky, sampprobsky, 
                          timestop=timestop)[[1]]
  }
  n <- tree$Nnode + 1
  heights <- dateNodes(tree)
  root_height <- max(heights)
  print(root_height)
  dates <- root_height - heights[1:n]
  
  for (j in 1:n){
    tree$tip.label[j] = paste(tree$tip.label[j], toString(dates[j]), sep="_")
  }
  write.tree(tree, file=paste("trees_constant/sim",i,".tree", sep=""))
  
  seq <- seqgen(opts=options, rooted.tree=tree)
  write_lines(seq, file=paste("seqs_constant/sim",i,".nexus", sep=""))
  return(tree)
}

trees <- lapply(1:10,simulate_tree)
ggtree(trees[[1]]) + theme_tree2() +geom_tiplab()

get_height <- function(tree){
  return(max(nodeHeights(tree)))
}