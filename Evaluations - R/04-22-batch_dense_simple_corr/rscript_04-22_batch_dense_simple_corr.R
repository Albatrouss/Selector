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

#read in all the logs from logs/
path <-"04-22-batch_dense_simple_corr/"
lognames <- list.files(path=path, pattern = "*.csv")
logs <- vector("list",length(lognames))
testinterval <- 0
trainingsteps <- 0
shared_exp <- vector("list",length(lognames))
mode <- vector("logical",length(lognames))
batch_size <- vector("logical",length(lognames))
i <- 1
for (logname in lognames){
  #read in the header for determining the testing circumstances
  print(i)
  #print(info$training_steps)
  info <- read.csv(file.path(path, logname), sep=";", skip = 0, nrows = 1, stringsAsFactors = FALSE)
  #print(info$training_steps)
  batch_size[i] <- info$batch_size
  mode[i] <- info$mode
  testinterval <- info$test_interval
  trainingsteps <- info$training_steps
  testepisodes <- info$test_episodes
  #read in the actual data
  logs[i] <- list(read.csv(file.path(path, logname), sep=";", skip = 6))
  header_of_train <-  read.csv(file.path(path, logname), sep=";", skip = 5, nrows = 1, header=FALSE, stringsAsFactors = FALSE)
  i <- i +1 
}
testcount <- trainingsteps / testinterval +1
testduration <- testepisodes
#split data into training and eval
trainlog <- vector("list",length(lognames))
evallog <- vector("list",length(lognames))
i = 1
for (log in logs){
  current <- logs[[i]]
  #convert stuff to numeric :D
  split <- split(current, current$eval)
  trainlog[i] <- list(split$train)
  trainlog[[i]]$agent_steps <- as.numeric(levels(trainlog[[i]]$agent_steps))[trainlog[[i]]$agent_steps]
  trainlog[[i]] <- subset(trainlog[[i]], agent_steps < trainingsteps)
  
  evallog[i] <- list(split$eval)
  evallog[[i]]$agent_steps <- as.numeric(levels(evallog[[i]]$agent_steps))[evallog[[i]]$agent_steps]
  evallog[[i]] <- subset(evallog[[i]], agent_steps < trainingsteps)
  evallog[[i]]$mode <- rep(mode[i],length( evallog[[i]]$entrynr))
  evallog[[i]]$batch_size <- rep(batch_size[i],length( evallog[[i]]$entrynr))
  
  colnames(trainlog[[i]]) <- header_of_train[1,]
  trainlog[[i]][10:13]<-NULL
  trainlog[[i]]$agent_steps <- as.numeric(levels(trainlog[[i]]$agent_steps))[trainlog[[i]]$agent_steps]
  trainlog[[i]]$entrynr <- as.numeric(levels(trainlog[[i]]$entrynr))[trainlog[[i]]$entrynr]
  trainlog[[i]]$envnr <- as.numeric(levels(trainlog[[i]]$envnr))[trainlog[[i]]$envnr]
  trainlog[[i]] <- replace_with_na(trainlog[[i]], replace = list(std_dev = -1, std_dev_punished = -1))
  trainlog[[i]]$mode <- rep(mode[i],length( trainlog[[i]]$entrynr))
  trainlog[[i]]$batch_size <- rep(batch_size[i],length( trainlog[[i]]$entrynr))
  i <- i + 1
}

for (i in 1:length(trainlog)){
trainlog[[i]] <- subset(trainlog[[i]], agent_steps < (trainingsteps+testinterval -1))
evallog[[i]] <- subset(evallog[[i]], agent_steps < (trainingsteps+testinterval - 1))
}


logs <- NULL
#bind them together
trainlogs <- bind_rows(trainlog, .id = "nr")
trainlogs$nr <- as.factor(as.numeric(trainlogs$nr))
trainlogs$mode <- as.factor(trainlogs$mode)
trainlogs$batch_size <- as.factor(trainlogs$batch_size)
#levels(trainlogs$epsilon)<- c("without epsilon-greedy", "with epsilon-greedy")
#plot th4 reward from training split by agent

evallogs <- bind_rows(evallog, .id = "nr")
evallogs$nr <- as.factor(as.numeric(evallogs$nr))
evallogs$mode <- as.factor(evallogs$mode)
evallogs$batch_size <- as.factor(evallogs$batch_size)

#testing percentages plotted as a distribution
#bunch them into distinct episodes and look at where correlations lie (reward and correctly selected (aka learning?) or epsilon ...)
accuracies <- vector(mode="list", length = length(evallog))
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
    std_r = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    max_std = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    max_pun = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    norm_std = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    norm_pun = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    x = (0:(testcount-1) * testinterval),
    m = rep(as.factor(mode[elogind]), ),
    b = rep(as.factor(batch_size[elogind]), ))
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
    
    #recalc to use just the average
    a0_std <- batch_$Agent0_std
    a1_std <- batch_$Agent1_std
    a2_std <- batch_$Agent2_std
    
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
    accuracy$std_r[k] <- mean(winner3 == batch$envnr)
    
    k <- k + 1
    j <- j + testduration
  }
  print(elogind)
  accuracies[elogind] <- list(accuracy)
}
#bind them together

totalacc <- bind_rows(accuracies, .id = "nr")

#cause they are 0 ... argh
totalacc <- subset(totalacc, totalacc$x < 1000000)

totalacc$nr <- as.factor(as.numeric(totalacc$nr))
totalacc$m <- as.factor((totalacc$m))
totalacc$b <- as.factor((totalacc$b))
#levels(totalacc$e)<- c("without epsilon-greedy", "with epsilon-greedy")
#graph these points
#learning curves of accuracy

trainlogs$envnr <- as.factor(trainlogs$envnr)
levels(trainlogs$envnr)[match("0",levels(trainlogs$envnr))] <- "Fruitbot"
levels(trainlogs$envnr)[match("1",levels(trainlogs$envnr))] <- "Starpilot"
levels(trainlogs$envnr)[match("2",levels(trainlogs$envnr))] <- "Coinrun"

levels(trainlogs$mode)[match("conv",levels(trainlogs$mode))] <- "Eine Schicht in Köpfen"
levels(trainlogs$mode)[match("conv2",levels(trainlogs$mode))] <- "Zwei Schichten in Köpfen"

levels(evallogs$mode)[match("conv",levels(evallogs$mode))] <- "Eine Schicht in Köpfen"
levels(evallogs$mode)[match("conv2",levels(evallogs$mode))] <- "Zwei Schichten in Köpfen"

levels(totalacc$m)[match("conv",levels(totalacc$m))] <- "Eine Schicht in Köpfen"
levels(totalacc$m)[match("conv2",levels(totalacc$m))] <- "Zwei Schichten in Köpfen"

#train reward ~ mode envnr
ggplot(trainlogs)+
  theme_light() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 1000000), breaks = seq(0, 1000000, by = 250000), labels = c("0","250k","500k","750k","1M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_grid(~trainlogs$envnr~ trainlogs$mode)+#~trainlogs$batch_size)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$reward, colour = trainlogs$nr), size = 0.1, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$reward), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte")+
  ylab("Belohnung")+
  theme(legend.position="none")

#train stddev ~mode envnr
ggplot(trainlogs)+
  theme_light() +
  #geom_point(size=1)+
  scale_y_continuous(limits=c(0,5))+
  scale_x_continuous(limits = c(0, 1000000), breaks = seq(0, 1000000, by = 250000), labels = c("0","250k","500k","750k","1M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_grid(~trainlogs$envnr~ trainlogs$mode)+#~trainlogs$batch_size)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$std_dev_punished, colour = trainlogs$nr), size = 0.125, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$std_dev_punished), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte")+
  ylab("Unsicherheitsmaß")+
  theme(legend.position="none")

#eval acc ~mode
ggplot(totalacc)+
  theme_light() +
  scale_x_continuous(limits = c(0, 1000000), breaks = seq(0, 1000000, by = 250000), labels = c("0","250k","500k","750k","1M"))+
  scale_y_continuous(limits = c(0,1))+
  facet_wrap(totalacc$m)+#~ totalacc$nr)+
  #geom_point(aes(x=x, y=totalacc$norm_pun, colour = nr), size=1)
  geom_smooth(aes(x=x,y=totalacc$std,  linetype = "Ohne Normierung", colour=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.90)+
  geom_smooth(aes(x=x,y=totalacc$std,  linetype = "Ohne Normierung"),method="auto", se=FALSE, fullrange=FALSE, level=0.90)+
  geom_smooth(aes(x=x,y=totalacc$norm_pun ,  linetype = "Normiert mit den letzten 500 Trainingswerten", colour=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.90)+
  geom_smooth(aes(x=x,y=totalacc$norm_pun ,  linetype = "Normiert mit den letzten 500 Trainingswerten"),method="auto", se=FALSE, fullrange=FALSE, level=0.90)+
  ylab("Anteil richtig ausgewählter Agenten")+
  xlab("Schritte")+
  guides(linetype=guide_legend(title=""))+
  theme(legend.position = 'bottom')+
  guides(colour=FALSE)


#eval reward mode
ggplot(evallogs)+
  theme_light() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 1000000), breaks = seq(0, 1000000, by = 250000), labels = c("0","250k","500k","750k","1M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_wrap(~evallogs$mode)+#~ evallogs$mode)+#~trainlogs$batch_size)+
  geom_smooth(aes(x=evallogs$agent_steps, y=evallogs$reward, colour = evallogs$nr), size = 0.1, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=evallogs$agent_steps, y=evallogs$reward), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte")+
  ylab("Belohnung")+
  theme(legend.position="none")







ggplot(totalacc)+
  theme_bw() +
  scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 4000000, by = 2000000))+
  facet_wrap(totalacc$s)+
  geom_point(aes(x=x, y=totalacc$norm_pun, colour = nr), size=1)+
  geom_smooth(aes(x=x,y=totalacc$pun),method="auto", se=TRUE, fullrange=FALSE, level=0.95,  colour = "black")+
  geom_smooth(aes(x=x,y=totalacc$norm_pun),method="auto", se=TRUE, fullrange=FALSE, level=0.95,linetype = "dashed", colour = "black")

