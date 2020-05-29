setwd("~/Dropbox/Uni/MAInformatik/R")
library(dplyr)
library(ggplot2)
library(naniar)
library(scales)


numerise <- function(x){
  if(!is.numeric(x)){
  n <- as.numeric(levels(x))[x]
  n}
  else{
    x
  }
}

#read in all the logs from logs/
path <-"05-20-vergleich-procgen-single-multi/"
lognames <- list.files(path=path, pattern = "*.csv")
logs <- vector("list",length(lognames))
testinterval <- 0
trainingsteps <- 0
shared_exp <- vector("list",length(lognames))
mode <- vector("logical",length(lognames))
batch_size <- vector("logical",length(lognames))
single_multi <- vector("logical",length(lognames))
i <- 1
for (logname in lognames){
  #read in the header for determining the testing circumstances
  print(i)
  #print(info$training_steps)
  info <- read.csv(file.path(path, logname), sep=";", skip = 0, nrows = 1, stringsAsFactors = FALSE)
  info2 <- read.csv(file.path(path, logname), sep=";", skip = 3, nrows = 1, stringsAsFactors = FALSE)
  shared_exp[i] <- info$shared_experience
  #print(info$training_steps)
  batch_size[i] <- info$batch_size
  mode[i] <- info$mode
  single_multi[i]<- info2$single.multi
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
  #evallog[[i]]$correctly_selected <- as.numeric(levels(evallog[[i]]$correctly_selected))[evallog[[i]]$correctly_selected]
  evallog[[i]]$single.multi <- rep(single_multi[i],length( evallog[[i]]$entrynr))
  evallog[[i]]$nr <- rep(i,length( evallog[[i]]$entrynr))
  #evallog[[i]]$correctly_selected <- as.numeric(levels(evallog[[i]]$correctly_selected))[evallog[[i]]$correctly_selected]
  #evallog[[i]] <- subset(evallog[[i]], agent_steps < trainingsteps)
  
  colnames(trainlog[[i]]) <- header_of_train[1,]
  trainlog[[i]][10:13]<-NULL
  trainlog[[i]]$agent_steps <- as.numeric(levels(trainlog[[i]]$agent_steps))[trainlog[[i]]$agent_steps]
  trainlog[[i]]$entrynr <- as.numeric(levels(trainlog[[i]]$entrynr))[trainlog[[i]]$entrynr]
  trainlog[[i]]$envnr <- as.numeric(levels(trainlog[[i]]$envnr))[trainlog[[i]]$envnr]
  trainlog[[i]] <- replace_with_na(trainlog[[i]], replace = list(std_dev = -1, std_dev_punished = -1))
  trainlog[[i]]$mode <- rep(mode[i],length( trainlog[[i]]$entrynr))
  trainlog[[i]]$batch_size <- rep(batch_size[i],length( trainlog[[i]]$entrynr))
  trainlog[[i]]$single.multi <- rep(single_multi[i],length( trainlog[[i]]$entrynr))
  i <- i + 1
}

#for (i in 1:length(trainlog)){
#trainlog[[i]] <- subset(trainlog[[i]], agent_steps < (trainingsteps+testinterval -1))
#evallog[[i]] <- subset(evallog[[i]], agent_steps < (trainingsteps+testinterval - 1))
#}


logs <- NULL
#bind them together
library(plyr)
evallogs <- rbind.fill(evallog, .id = "nr")
#evallogs <- bind_rows(evallog, .id = "nr")

evallogs$nr <- as.factor(as.numeric(evallogs$nr))
#evallogs$mode <- as.factor(evallogs$mode)
#evallogs$batch_size <- as.factor(evallogs$batch_size)
#evallogs$single.multi <- as.numeric(as.character(evallogs$single.multi))
levels(evallogs$envnr)[match("0",levels(evallogs$envnr))] <- "fruitbot"
levels(evallogs$envnr)[match("1",levels(evallogs$envnr))] <- "bigfish"
levels(evallogs$envnr)[match("2",levels(evallogs$envnr))] <- "climber"

evallogs$envnr <- factor(evallogs$envnr)


trainlogs <- bind_rows(trainlog, .id = "nr")
trainlogs$nr <- as.factor(as.numeric(trainlogs$nr))
trainlogs$mode <- as.factor(trainlogs$mode)
trainlogs$batch_size <- as.factor(trainlogs$batch_size)
trainlogs$single.multi <- as.factor(trainlogs$single.multi)


split <- split(trainlogs, trainlogs$single.multi)
single <- split$single
single$agent_steps <- single$agent_steps /3
trainlogs <- bind_rows(single, split$multi)


split <- split(evallogs, evallogs$single.multi)
single <- split$single
single$agent_steps <- single$agent_steps /3
evallogs <- bind_rows(single, split$multi)



#levels(trainlogs$epsilon)<- c("without epsilon-greedy", "with epsilon-greedy")
#plot th4 reward from training split by agent
evallogs$correctly_selected <- as.numeric(evallogs$correctly_selected)


trainlogs$envnr[trainlogs$envnr == '0'] <- "fruitbot"
trainlogs$envnr[trainlogs$envnr == '1'] <- "bigfish"
trainlogs$envnr[trainlogs$envnr == '2'] <- "climber"

levels(trainlogs$single.multi)[match("single",levels(trainlogs$single.multi))] <- "Selector mit einem Agenten"
levels(trainlogs$single.multi)[match("multi",levels(trainlogs$single.multi))] <- "Selector mit drei Agenten"

#trainlogs$single.multi[trainlogs$single.multi == 'single'] <- "Einzelner Agent"
#trainlogs$single.multi[trainlogs$single.multi == 'multi'] <- "Mehrere Agenten"


#reward_by_environment_training 
ggplot(trainlogs)+#, colour = trainlogs$nr))+
  theme_light() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000), labels = c("0","1M","2M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_wrap(~trainlogs$envnr)+
  geom_smooth( aes(x=trainlogs$agent_steps, y=trainlogs$reward, colour=trainlogs$single.multi), method="auto",size=1.5, se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth( aes(x=trainlogs$agent_steps, y=trainlogs$reward, colour=trainlogs$single.multi, fill=trainlogs$nr), size=0.2, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Schritte in Umgebung")+
  ylab("Belohnung")+ 
  guides(colour=guide_legend(title="")) +
  theme(legend.position = 'bottom')+
  guides(fill=FALSE)
  #theme(axis.text.x = element_text(angle=45))

multi <- split(evallogs, evallogs$single.multi)
multi <- multi$multi
multi$Agent0_std_punished <- as.numeric(as.character(multi$Agent0_std_punished))
multi$Agent1_std_punished <- as.numeric(as.character(multi$Agent1_std_punished))
multi$Agent2_std_punished <- as.numeric(as.character(multi$Agent2_std_punished))
#eval stddev multi only intra environments
ggplot(multi)+
  theme_light() +
  #geom_point(size=0.05)+
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000), labels = c("0","1M","2M"))+
  scale_y_continuous(limits = c(0,2))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_wrap(~ multi$envnr)+
  #geom_line(aes(y=rollmean(evallogs$correctly_selected,100,na.pad=TRUE)))+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent0_std_punished, colour="fruitbot"),method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent1_std_punished, colour="bigfish"),method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent2_std_punished, colour="climber"),method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent0_std_punished, colour="fruitbot", fill=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent1_std_punished, colour="bigfish",fill=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent2_std_punished, colour="climber",fill=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  xlab("Schritte pro Umgebung")+ 
  guides(colour=guide_legend(title="Agent")) +
  theme(legend.position = 'bottom')+
  guides(fill=FALSE)+
  ylab("Unsicherheitsmaß")

#reward_total_training <- 
ggplot(evallogs)+
  theme_light() +
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000), labels = c("0","1M","2M"))+
  geom_smooth(aes(x=evallogs$agent_steps, y=evallogs$reward, colour=single.multi),size=2, method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=evallogs$agent_steps, y=evallogs$reward, colour=single.multi, fill=nr),size=0.1, method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  xlab("Schritte pro Umgebung")+ 
  ylab("Belohnung")+
  guides(colour=guide_legend(title="Modell")) +
  theme(legend.position = 'bottom')+
  guides(fill=FALSE)


split <- split(evallogs, evallogs$single.multi)
multi <- split$multi

reward_by_environment_testing_multi <-ggplot(multi, aes(x=multi$agent_steps, y=multi$reward, colour=nr))+
  theme_bw() +
  #geom_point(size=0.05)+
  facet_wrap(~multi$envnr)+
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000), labels = c("0","1M","2M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  #facet_wrap(~ evallogs$single.multi)+
  #geom_line(aes(y=rollmean(evallogs$correctly_selected,100,na.pad=TRUE)))+
  geom_smooth( method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  xlab("Schritte pro Umgebung")+ 
  guides(colour=guide_legend(title="Model")) +
  ylab("Belohnung")
reward_by_environment_testing_multi

reward_by_environment_testing <-ggplot(evallogs, aes(x=evallogs$agent_steps, y=evallogs$reward, colour=single.multi))+
  theme_bw() +
  #geom_point(size=0.05)+
  facet_wrap(~evallogs$envnr)+
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000), labels = c("0","1M","2M"))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  #facet_wrap(~ evallogs$single.multi)+
  #geom_line(aes(y=rollmean(evallogs$correctly_selected,100,na.pad=TRUE)))+
  geom_smooth( method="auto", se=TRUE, fullrange=FALSE, level=0.95)+
  xlab("Schritte pro Umgebung")+ 
  guides(colour=guide_legend(title="Model")) +
  ylab("Belohnung")
reward_by_environment_testing

#std deviation split on env and shared_Exp
ggplot(trainlogs, aes(x=trainlogs$agent_steps, y=trainlogs$std_dev))+#, colour = trainlogs$nr))+
  theme_bw() +
  #geom_point(size=1)+
  scale_x_continuous(limits = c(0, 1000000), breaks = seq(0, 1000000, by = 500000))+
  scale_y_continuous(limits = c(0, 10))+
  #scale_y_log10(limits = c(0.001, 100))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_wrap(~trainlogs$envnr~ trainlogs$mode ~trainlogs$batch_size)+
  geom_smooth( method="auto", se=FALSE, fullrange=FALSE, level=0.99)+  
  xlab("steps")+
  ylab("std_dev")

#plot stddev and reward together
ggplot(trainlogs, aes(x=trainlogs$agent_steps, y=trainlogs$reward))+
  theme_bw() +
  #geom_point(size=1)+
  #scale_x_continuous(limits = c(0, 4000000), breaks = seq(0, 1000000, by = 500000))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_wrap(~trainlogs$mode~trainlogs$envnr~trainlogs$batch_size)+
  #scale_y_log10(limits = c(0.1, 10))+
  scale_y_continuous(limits = c(-2.5, 5),  "reward",     sec.axis = sec_axis(~ ., name = "std. deviation")  )+
  geom_smooth( aes(x=trainlogs$agent_steps, y=trainlogs$reward, colour = trainlogs$nr, linetype="Reward"), method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=trainlogs$agent_steps, y=trainlogs$std_dev,colour = trainlogs$nr, linetype="Standard deviation"), method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  xlab("steps")#+
  #ylab("rewar std. dev")
#scale_colour_manual(values = c('1' = "yellow", "12" = "red"))#TODO doesn't work just yet


#testing percentages plotted as a distribution
#bunch them into distinct episodes and look at where correlations lie (reward and correctly selected (aka learning?) or epsilon ...)
accuracies <- vector(mode="list", length = length(evallog))
for (elogind in 1:length(evallog)) {
  print(elogind)
  elog <- evallog[[elogind]]
  nr <- elog$nr[1]
  print(nr)
  if (elog$single.multi[1] == "single"){
    print("single")
  }
  else{
  accuracy <- data.frame(
    nr = rep(nr, length(as.integer(length(elog$correctly_selected)/testduration))),
    std = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    pstd = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    norm_pun = vector("logical", length = length(as.integer(length(elog$correctly_selected)/testduration))),
    x = (0:(testcount-1) * testinterval))
  trainsplit <- split(trainlog[[elogind]], trainlog[[elogind]]$envnr, drop=TRUE)

  #recalculate the result for every entry
  
  #calculate a second accuracy: accuracy_normed_100
  
  j <- 1 #start
  k<-1 # index
  while (j < length(elog$correctly_selected)) {
    end <- j + testduration -1
    batch <- elog[j:end,]
    accuracy$norm_pun[k] <- mean(batch$correctly_selected)
    #rescale all std and std pun
    batch_ <- lapply(batch, numerise)

    a0_std <- batch_$Agent0_std
    a1_std <- batch_$Agent1_std
    a2_std <- batch_$Agent2_std
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
    accuracy$std[k] <- mean(winner == batch$envnr)
    
   
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
    accuracy$pstd[k] <- mean(winner3 == batch$envnr)
    
    k <- k + 1
    j <- j + testduration
  }
  print(elogind)
  accuracies[elogind] <- list(accuracy)
}}
#bind them together

totalacc <- rbind.fill(accuracies, .id = "nr")

#cause they are 0 ... argh
#totalacc <- subset(totalacc, totalacc$x < 1000000)

totalacc$nr <- as.factor(as.numeric(totalacc$nr))
#totalacc$m <- as.factor((totalacc$m))
#totalacc$b <- as.factor((totalacc$b))
#levels(totalacc$e)<- c("without epsilon-greedy", "with epsilon-greedy")
#graph these points
#learning curves of accuracy
ggplot(totalacc)+
  theme_light() +
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000), labels = c("0","1M","2M"))+
  scale_y_continuous(limits = c(0,1))+
  geom_smooth(aes(x=x,y=totalacc$norm_pun,  colour = "Normiert über 500 Episoden"),size=2,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=x,y=totalacc$pstd, colour="Nicht Normiert"), size=2, method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=x,y=totalacc$norm_pun,  colour = "Normiert über 500 Episoden", fill=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=x,y=totalacc$pstd, colour="Nicht Normiert",fill=nr),size=0.1, method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  ylab("Anteil richtiger Auswahlen")+
  xlab("Schritte")+
  theme(legend.position = 'bottom')+
  guides(colour=guide_legend(title=""))+
  guides(fill=FALSE)

  
ggplot(multi)+
  theme_light() +
  #geom_point(size=0.05)+
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000), labels = c("0","1M","2M"))+
  scale_y_continuous(limits = c(0,2))+
  #geom_point(size=1, aes(y = trainlog[[i]]$std_dev, colour="std"))+
  facet_wrap(~ multi$envnr)+
  #geom_line(aes(y=rollmean(evallogs$correctly_selected,100,na.pad=TRUE)))+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent0_std_punished, colour="fruitbot"),size=1.5,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent1_std_punished, colour="bigfish"),size=1.5,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent2_std_punished, colour="climber"),size=1.5,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent0_std_punished, colour="fruitbot", fill=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent1_std_punished, colour="bigfish",fill=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth( aes(x=multi$agent_steps, y=multi$Agent2_std_punished, colour="climber",fill=nr),size=0.1,method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  xlab("Schritte pro Umgebung")+ 
  guides(colour=guide_legend(title="Modell")) +
  theme(legend.position = 'bottom')+
  guides(fill=FALSE)+
  ylab("Unsicherheitsmaß")

ggplot(totalacc)+
  theme_bw() +
  scale_x_continuous(limits = c(0, 2000000), breaks = seq(0, 2000000, by = 1000000))+
  scale_y_continuous(limits = c(0,1))+
  #facet_wrap(totalacc$m ~totalacc$b)+#~ totalacc$nr)+
  #geom_point(aes(x=x, y=totalacc$norm_pun, colour = nr), size=1)
  geom_smooth(aes(x=x,y=totalacc$std,  colour = "using minimum standard deviation"),method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=x,y=totalacc$norm_pun ,  colour = "Normed using mean over last 500 trainingepisodes"),method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  #geom_smooth(aes(x=x,y=totalacc$norm_pun ,  linetype = "Normed using mean over last 500 trainingepisodes", colour = nr),size=0.25, method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=x,y=totalacc$pstd ,  colour = "using minimum punished standard deviation"),method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  ylab("Anteil richtiger Vorhersagen")+
  xlab("Schritte")
  #+theme(legend.position="none")

