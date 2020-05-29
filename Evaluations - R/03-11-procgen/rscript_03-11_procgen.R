setwd("~/Dropbox/Uni/MAInformatik/R")
library(dplyr)
library(ggplot2)
library(naniar)

numerise <- function(x){
  if(!is.numeric(x)){
  n <- as.numeric(levels(x))[x]
  n}
  else{
    x
  }
}
#0 chaser
#1 bigfish
#2 caveflyer


#read in all the logs from logs/
path <-"03-11-procgen/"
lognames <- list.files(path=path, pattern = "*.csv")
logs <- vector("list",length(lognames))
testinterval <- 0
trainingsteps <- 0
shared_exp <- vector("list",length(lognames))
i <- 1
for (logname in lognames){
  #read in the header for determining the testing circumstances
  info <- read.csv(file.path(path, logname), sep=";", skip = 2, nrows = 1, stringsAsFactors = FALSE)
  shared_exp[i] <- info$shared_experience
  testinterval <- info$test_interval
  trainingsteps <- info$training_steps
  #read in the actual data
  logs[i] <- list(read.csv(file.path(path, logname), sep=";", skip = 5))
  header_of_train <-  read.csv(file.path(path, logname), sep=";", skip = 4, nrows = 1, header=FALSE, stringsAsFactors = FALSE)
  i <- i +1 
}
testcount <- trainingsteps / testinterval
testduration <- 50
#split data into training and eval
trainlog <- vector("list",length(lognames))
evallog <- vector("list",length(lognames))
i = 1
for (log in logs){
  print(i)
  current <- logs[[i]]
  #convert stuff to numeric :D
  split <- split(current, current$eval)
  trainlog[i] <- list(split$train)
  evallog[i] <- list(split$eval)
  evalsteps <- sort(rep((1:testcount * testinterval), testduration))
  evallog[[i]]$agent_steps <- evalsteps
  evallog[[i]]$entrynr <- as.numeric(levels(evallog[[i]]$entrynr))[evallog[[i]]$entrynr]
  evallog[[i]]$envnr <- as.numeric(levels(evallog[[i]]$envnr))[evallog[[i]]$envnr]
  evallog[[i]]$shared_experience <- rep(shared_exp[i],length( evallog[[i]]$entrynr))
  
  
  colnames(trainlog[[i]]) <- header_of_train[1,]
  trainlog[[i]][10:13]<-NULL
  trainlog[[i]]$agent_steps <- as.numeric(levels(trainlog[[i]]$agent_steps))[trainlog[[i]]$agent_steps]
  trainlog[[i]]$entrynr <- as.numeric(levels(trainlog[[i]]$entrynr))[trainlog[[i]]$entrynr]
  trainlog[[i]]$envnr <- as.numeric(levels(trainlog[[i]]$envnr))[trainlog[[i]]$envnr]
  trainlog[[i]] <- replace_with_na(trainlog[[i]], replace = list(std_dev = -1, std_dev_punished = -1))
  trainlog[[i]]$shared_experience <- rep(shared_exp[i],length( trainlog[[i]]$entrynr))
  i <- i + 1
}
#bind them together
evallogs <- bind_rows(evallog, .id="nr")
evallogs$nr <- as.factor(as.numeric(evallogs$nr))
evallogs$shared_experience <- as.factor(as.numeric(evallogs$shared_experience))

trainlogs <- bind_rows(trainlog, .id = "nr")
trainlogs$nr <- as.factor(as.numeric(trainlogs$nr))
trainlogs$shared_experience <- as.factor(as.numeric(trainlogs$shared_experience))
#levels(trainlogs$epsilon)<- c("without epsilon-greedy", "with epsilon-greedy")
#plot th4 reward from training split by agent

#testing percentages plotted as a distribution
#bunch them into distinct episodes and look at where correlations lie (reward and correctly selected (aka learning?) or epsilon ...)
accuracies <- vector(mode="list", length = length(evallog))
maxs_std <- vector(mode="list", length = length(evallog))
maxs_pun <- vector(mode="list", length = length(evallog))
normfacs_std <- vector(mode="list", length = length(evallog))
normfacs_pun <- vector(mode="list", length = length(evallog))
for (elogind in 1:length(evallog)) {
  print(elogind)
  elog <- evallog[[elogind]]
  accuracy <- data.frame(
    std = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    pun = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    max_std = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    max_pun = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    norm_std = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    norm_pun = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    x = (1:testcount * testinterval),
    s = rep(as.numeric(shared_exp[elogind]), ))
  trainsplit <- split(trainlog[[elogind]], trainlog[[elogind]]$envnr, drop=TRUE)
  normfac_std = matrix(nrow = length(trainsplit), ncol = testcount)
  normfac_pun = matrix(nrow = length(trainsplit), ncol = testcount)
  #max for oodd
  max_std = matrix(nrow = length(trainsplit), ncol = testcount)
  max_pun = matrix(nrow = length(trainsplit), ncol = testcount)
  
  trainsplit <- split(trainlog[[elogind]], trainlog[[elogind]]$envnr, drop=TRUE)
  for (env in trainsplit){
    lastnr <- 0
    envnr <- env$envnr[1] + 1
    for (currentnr in accuracy$x){
      sample <- tail(subset(env, (agent_steps > lastnr & agent_steps < currentnr), drop=TRUE),500)
      max_std[envnr, as.integer(currentnr/testinterval)] <- summary(sample$std_dev)[[5]]
      max_pun[envnr, as.integer(currentnr/testinterval)] <-  summary(sample$std_dev_punished)[[5]]
      normfac_std[envnr, as.integer(currentnr/testinterval)] <- mean(sample$std_dev,  na.rm =TRUE)
      normfac_pun[envnr, as.integer(currentnr/testinterval)] <- mean(sample$std_dev_punished,  na.rm =TRUE)
      lastnr <- currentnr
    }
  }
  maxs_std[[elogind]]<-max_std
  maxs_pun[[elogind]]<-max_pun
  normfacs_std[[elogind]]<-normfac_std
  normfacs_pun[[elogind]]<-normfac_pun
  
  #recalculate the result for every entry
  
  #calculate a second accuracy: accuracy_normed_100
  
  j <- 1 #start
  k<-1 # index
  while (j < length(elog$correctly_selected)) {
    end <- j + testduration -1
    batch <- elog[j:end,]
    accuracy$std[k] <- mean(batch$correctly_selected)
    #rescale all std and std pun
    batch$agent_steps <- NULL
    batch_ <- lapply(batch, numerise)
    
    norm_std <- function(x, normer){
       a <- x / normer
       a
    }
    isovermax_stds <- vector("list", length = length(batch$envnr))
    isovermax_puns <- vector("list", length = length(batch$envnr))
    for(i in 1:length(batch$envnr)){
      #check whether this is over the max for all 6
      isovermax_std <- vector("logical",length = length(trainsplit))
      isovermax_pun <- vector("logical",length = length(trainsplit))
      
      isovermax_std[1] <-batch_$Agent0_std[i] < max_std[1,k]
      isovermax_std[2] <-batch_$Agent1_std[i] < max_std[2,k]
      isovermax_std[3] <-batch_$Agent2_std[i] < max_std[3,k]
      
      isovermax_pun[1] <-batch_$Agent0_std_punished[i] < max_pun[1,k]
      isovermax_pun[2] <-batch_$Agent1_std_punished[i] < max_pun[2,k]
      isovermax_pun[3] <-batch_$Agent2_std_punished[i] < max_pun[3,k]
      
      isovermax_stds[i] <- list(isovermax_std)
      isovermax_puns[i] <- list(isovermax_pun)
      
    }
    
    #calc winner for std and punished excluding "outliers"
    
    
    a0_std <- batch_$Agent0_std
    a1_std <- batch_$Agent1_std
    a2_std <- batch_$Agent2_std
    
    winner<- vector("logical",length=length(a0_std))
    
    for(i in 1:length(a0_std)){
      if((all(isovermax_stds[[i]])||!isovermax_stds[[i]][1])&& a0_std[i] < a1_std[i] && a0_std[i] < a2_std[i]){
        winner[i] <- 0
      }
      if((all(isovermax_stds[[i]])||!isovermax_stds[[i]][2])&&a1_std[i] < a0_std[i] && a1_std[i] < a2_std[i]){
        winner[i] <- 1
      }
      if((all(isovermax_stds[[i]])||!isovermax_stds[[i]][3])&&a2_std[i] < a1_std[i] && a2_std[i] < a0_std[i]){
        winner[i] <- 2
      }
    }
    accuracy$max_std[k]  <- mean(winner == batch$envnr)
    
    
    a0_std <- batch_$Agent0_std_punished
    a1_std <- batch_$Agent1_std_punished
    a2_std <- batch_$Agent2_std_punished
    
    winner<- vector("logical",length=length(a0_std))
    
    for(i in 1:length(a0_std)){
      if((all(isovermax_puns[[i]])||!isovermax_puns[[i]][1])&& a0_std[i] < a1_std[i] && a0_std[i] < a2_std[i]){
        winner[i] <- 0
      }
      if((all(isovermax_puns[[i]])||!isovermax_puns[[i]][2])&&a1_std[i] < a0_std[i] && a1_std[i] < a2_std[i]){
        winner[i] <- 1
      }
      if((all(isovermax_puns[[i]])||!isovermax_puns[[i]][3])&&a2_std[i] < a1_std[i] && a2_std[i] < a0_std[i]){
        winner[i] <- 2
      }
    }
    accuracy$max_pun[k]  <- mean(winner == batch$envnr)
    
    
    
    
    #todo: write it to batch
    a0_std <- norm_std(batch_$Agent0_std, normfac_std[1,k])
    a1_std <- norm_std(batch_$Agent1_std, normfac_std[2,k])
    a2_std <- norm_std(batch_$Agent2_std, normfac_std[3,k])
    #calculate winner
    winner <- vector("logical",length=length(a0_std))
    for(i in 1:length(a0_std)){
      if(a0_std[i] < a1_std[i] && a0_std[i] < a2_std[i]){
        winner[i] <- 0
      }
      if(a1_std[i] < a0_std[i] && a1_std[i] < a2_std[i]){
        winner[i] <- 1
      }
      if(a2_std[i] < a1_std[i] && a2_std[i] < a0_std[i]){
        winner[i] <- 2
      }
    }
    accuracy$norm_std[k] <- mean(winner == batch$envnr)

    #reuse names for convenience, compute punished average
    a0_std <- norm_std(batch_$Agent0_std_punished, normfac_pun[1,k])
    a1_std <- norm_std(batch_$Agent1_std_punished, normfac_pun[2,k])
    a2_std <- norm_std(batch_$Agent2_std_punished, normfac_pun[3,k])
    
    winner2 <- vector("logical",length=length(a0_std))
    for(i in 1:length(a0_std)){
      if(a0_std[i] < a1_std[i] && a0_std[i] < a2_std[i]){
        winner2[i] <- 0
      }
      if(a1_std[i] < a0_std[i] && a1_std[i] < a2_std[i]){
        winner2[i] <- 1
      }
      if(a2_std[i] < a1_std[i] && a2_std[i] < a0_std[i]){
        winner2[i] <- 2
      }
    }
    accuracy$norm_pun[k] <- mean(winner2 == batch$envnr)
    
    #recalc to use just the punished average
    a0_std <- batch_$Agent0_std_punished
    a1_std <- batch_$Agent1_std_punished
    a2_std <- batch_$Agent2_std_punished
    
    winner3 <- vector("logical",length=length(a0_std))
    for(i in 1:length(a0_std)){
      if(a0_std[i] < a1_std[i] && a0_std[i] < a2_std[i]){
        winner3[i] <- 0
      }
      if(a1_std[i] < a0_std[i] && a1_std[i] < a2_std[i]){
        winner3[i] <- 1
      }
      if(a2_std[i] < a1_std[i] && a2_std[i] < a0_std[i]){
        winner3[i] <- 2
      }
    }
    accuracy$pun[k] <- mean(winner3 == batch$envnr)
  
    k <- k + 1
    j <- j + testduration
  }
  print(elogind)
  accuracies[elogind] <- list(accuracy)
}

#bind them together
totalacc <- bind_rows(accuracies, .id = "nr")
totalacc$nr <- as.factor(as.numeric(totalacc$nr))
totalacc$s <- as.factor(as.numeric(totalacc$s))

trainlogs$envnr <- as.factor(trainlogs$envnr)
levels(trainlogs$envnr)[match("0",levels(trainlogs$envnr))] <- "Chaser"
levels(trainlogs$envnr)[match("1",levels(trainlogs$envnr))] <- "Bigfish"
levels(trainlogs$envnr)[match("2",levels(trainlogs$envnr))] <- "Caveflyer"
evallogs$entrynr <- as.factor(as.numeric(evallogs$entrynr))
evallogs$envnr <- as.factor(evallogs$envnr)
levels(evallogs$envnr)[match("0",levels(evallogs$envnr))] <- "Chaser"
levels(evallogs$envnr)[match("1",levels(evallogs$envnr))] <- "Bigfish"
levels(evallogs$envnr)[match("2",levels(evallogs$envnr))] <- "Caveflyer"

evallogs$Agent0_std_punished <- as.numeric(evallogs$Agent0_std_punished)
evallogs$Agent1_std_punished <- as.numeric(evallogs$Agent1_std_punished)
evallogs$Agent2_std_punished <- as.numeric(evallogs$Agent2_std_punished)


#learning curves according to environment and shared experience
ggplot(trainlogs)+
  theme_light() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000), labels = c("0","2M","4M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_grid(~trainlogs$envnr~ trainlogs$shared_experience)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$reward, colour = trainlogs$nr), size=0.1, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$reward), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte")+
  ylab("Belohnung")+
  ggtitle("geteilte Erfahrung")+
  theme(plot.title = element_text(face="plain",size=11, hjust=0.5))+
  theme(legend.position="none")

#stddev according to env and shared exp
ggplot(trainlogs)+
  theme_light() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000), labels = c("0","2M","4M"))+
  scale_y_continuous(limits = c(0, 10))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_grid(~trainlogs$envnr~ trainlogs$shared_experience)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$std_dev_punished, colour = trainlogs$nr),size=0.1, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$std_dev_punished), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte")+
  ylab("Unsicherheitsmaß")+
  ggtitle("geteilte Erfahrung")+
  theme(plot.title = element_text(face="plain",size=11, hjust=0.5))+
  theme(legend.position="none")

#plot stddev and reward together NOT NEEDED
ggplot(trainlogs, aes(x=trainlogs$agent_steps, y=trainlogs$reward, colour = trainlogs$nr))+
  theme_bw() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_wrap(~trainlogs$shared_experience~trainlogs$envnr)+
  #scale_y_log10(limits = c(0.1, 10))+
  scale_y_continuous(limits = c(0, 3))+
  geom_smooth( method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$std_dev),linetype = "dashed", method="auto", se=TRUE, fullrange=FALSE, level=0.95)+
  xlab("steps")+
  ylab("reward")
#scale_colour_manual(values = c('1' = "yellow", "12" = "red"))#TODO doesn't work just yet

#learning curves of accuracy
ggplot(totalacc)+
  theme_light() +
  facet_wrap(totalacc$s)+
  #geom_point(aes(x=x, y=totalacc$std, colour = nr), size=1)+
  geom_smooth(aes(x=x,y=totalacc$std),method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=x,y=totalacc$std, colour = nr),method="auto",size=0.25, se=FALSE, fullrange=FALSE, level=0.99)+
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000), labels = c("0","2M","4M"))+
  scale_y_continuous(limits=c(0,1))+
  xlab("Schritte")+
  ylab("Anteil richtig ausgewählter Agenten")+
  ggtitle("geteilte Erfahrung")+
  theme(plot.title = element_text(face="plain",size=11, hjust=0.5))+
  theme(legend.position="none")


#learning curves according to environment and shared experience
ggplot(evallogs)+
  theme_light() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000), labels = c("0","2M","4M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_grid(~evallogs$envnr ~evallogs$shared_experience)+
  geom_smooth(aes(x=evallogs$agent_steps, y=evallogs$reward, colour = evallogs$nr),size=0.1, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=evallogs$agent_steps, y=evallogs$reward), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte")+
  ylab("Belohnung")+
  ggtitle("geteilte Erfahrung")+
  theme(plot.title = element_text(face="plain",size=11, hjust=0.5))+
  theme(legend.position="none")

#learning curves according to environment and shared experience
ggplot(evallogs)+
  theme_light() +
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000), labels = c("0","2M","4M"))+
  scale_y_continuous(limits=c(0,.5))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_grid(~evallogs$envnr~evallogs$shared_experience)+
  geom_smooth(aes(x=evallogs$agent_steps,y=evallogs$Agent0_std_punished, colour="Agent Chaser"), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=evallogs$agent_steps,y=evallogs$Agent1_std_punished, colour="Agent Bigfish"), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=evallogs$agent_steps,y=evallogs$Agent2_std_punished, colour="Agent Caveflyer"), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte")+
  ylab("Unsicherheitsmaß")+
  ggtitle("geteilte Erfahrung")+
  theme(plot.title = element_text(face="plain",size=11, hjust=0.5))+
  guides(colour=guide_legend(title=""))+
  theme(legend.position = 'bottom')
  #theme(legend.position="none")





ggplot(totalacc)+
  theme_bw() +
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000))+
  facet_wrap(totalacc$s)+
  #geom_point(aes(x=x, y=totalacc$norm_pun, colour = nr), size=1)+
  geom_smooth(aes(x=x,y=totalacc$pun, linetype="pun"),method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=x,y=totalacc$norm_pun, linetype = "normpun"),method="auto", se=FALSE, fullrange=FALSE, level=0.99)




