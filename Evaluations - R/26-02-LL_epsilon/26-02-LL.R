setwd("~/Dropbox/Uni/MAInformatik/R")
library(plyr)
library(dplyr)
library(ggplot2)
library(scales)
library(naniar)
#read in all the logs from logs/
path <-"26-02-LL_epsilon/"
lognames <- list.files(path=path, pattern = "*.csv")
logs <- vector("list",length(lognames))
shared_exp <- vector("list",length(lognames))
epsilon <- vector("logical",length(lognames))
i <- 1
for (logname in lognames){
  #read in the header for determining the testing circumstances
  info <- read.csv(file.path(path, logname), sep=";", skip = 2, nrows = 1)
  epsilon[i] <- info$epsilon.greedy
  shared_exp[i] <- info$shared_experience
  #read in the actual data
  logs[i] <- list(read.csv(file.path(path, logname), sep=";", skip = 5, stringsAsFactors = FALSE))
  header_of_train <-  read.csv(file.path(path, logname), sep=";", skip = 4, nrows = 1, header = FALSE,stringsAsFactors = FALSE)
  i <- i +1 
}
trainlog <- vector("list",length(lognames))
evallog <- vector("list",length(lognames))


i <- 1
for (log in logs){
  log <- logs[[i]]
  splitted <- split(log, log$eval)
  trainlog[i] <- list(splitted$train)
  colnames(trainlog[[i]]) <- header_of_train[1,]
  trainlog[[i]] <- subset(trainlog[[i]],select=c(entrynr, train, agentnr, envnr, agent_steps, std_dev_punished, std_dev, reward))
  trainlog[[i]] <- replace_with_na(trainlog[[i]], replace = list(std_dev = -1, std_dev_punished = -1))
  trainlog[[i]]$nr <- rep(i, length(trainlog[[i]]$entrynr))
  trainlog[[i]]$epsilon <- rep(epsilon[[i]], length(trainlog[[i]]$entrynr))
  
  evallog[i] <- list(splitted$eval)
  evallog[[i]]$nr <- rep(i, length(evallog[[i]]$entrynr))
  evallog[[i]]$epsilon <- rep(epsilon[[i]], length(evallog[[i]]$entrynr))
  
  i <- i + 1
}
trainbind <- rbind.fill(trainlog)
trainbind$entrynr <-  as.numeric(as.character(trainbind$entrynr))
trainbind$reward <-  as.numeric(as.character(trainbind$reward))
trainbind$std_dev <-  as.numeric(as.character(trainbind$std_dev))
trainbind$nr <- as.factor(trainbind$nr)
trainbind$envnr <- as.factor(trainbind$envnr)
trainbind$epsilon <- as.factor(trainbind$epsilon)
levels(trainbind$epsilon)[match("0",levels(trainbind$epsilon))] <- "Ohne Epsilon-greedy"
levels(trainbind$epsilon)[match("1",levels(trainbind$epsilon))] <- "Mit Epsilon-greedy"
levels(trainbind$envnr)[match("0",levels(trainbind$envnr))] <- "Landeplatz auf linker Seite"
levels(trainbind$envnr)[match("1",levels(trainbind$envnr))] <- "Landeplatz auf rechter Seite"

evalbind <- rbind.fill(evallog)
evalbind$entrynr <-  as.numeric(as.character(evalbind$entrynr))
evalbind$reward <-  as.numeric(as.character(evalbind$reward))
evalbind$Agent0_std_punished <-  as.numeric(as.character(evalbind$Agent0_std_punished))
evalbind$Agent1_std_punished <-  as.numeric(as.character(evalbind$Agent1_std_punished))
evalbind$envnr <- as.factor(evalbind$envnr)
levels(evalbind$envnr)[match("0",levels(evalbind$envnr))] <- "Landeplatz auf linker Seite"
levels(evalbind$envnr)[match("1",levels(evalbind$envnr))] <- "Landeplatz auf rechter Seite"
evalbind$nr <- as.factor(evalbind$nr)
evalbind$epsilon <- as.factor(evalbind$epsilon)
levels(evalbind$epsilon)[match("0",levels(evalbind$epsilon))] <- "Ohne Epsilon-greedy"
levels(evalbind$epsilon)[match("1",levels(evalbind$epsilon))] <- "Mit Epsilon-greedy"


trainbind$envnr <- as.factor(trainbind$envnr)


#reward training
ggplot(trainbind)+
  theme_light()+
  geom_smooth(aes(x=trainbind$entrynr, y=trainbind$reward, colour = trainbind$envnr),  size = 1.5, method="auto", se=TRUE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=trainbind$entrynr, y=trainbind$reward, colour = trainbind$envnr, fill=nr),  size = 0.25, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  xlab("Episode")+
  facet_wrap(~epsilon)+
  ylab("Belohnung")+
  #scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(limits = c(0,7000))+
  guides(colour=guide_legend(title="Umgebung"))+
  guides(fill=FALSE)+
  theme(legend.position = 'bottom')
#theme(legend.position="none")

#stddev training
ggplot(trainbind, aes(x=trainbind$entrynr, y=trainbind$std_dev))+#, colour = trainbind$nr))+
  theme_light()+
  geom_smooth(aes(x=trainbind$entrynr, y=trainbind$std_dev, colour = nr),size = 0.125, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=trainbind$entrynr, y=trainbind$std_dev),  size = 1, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  facet_grid(~envnr~epsilon)+
  xlab("Episode")+
  ylab("Unsicherheitsmaß")+
  #scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(limits = c(0,7000))+
  theme(legend.position="none")



#reward eval
ggplot(evalbind)+
  theme_light()+
  geom_smooth(aes(x=evalbind$entrynr, y=evalbind$reward, colour = evalbind$nr),  size = 0.25, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=evalbind$entrynr, y=evalbind$reward),  size = 2, method="auto", se=TRUE, fullrange=FALSE, level=0.95)+
  facet_wrap(~epsilon)+
  xlab("Episode")+
  ylab("Belohnung")+
  #scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(limits = c(0,7000))+
  theme(legend.position="none")

#stddev bei eval Vergleich
ggplot(evalbind)+
  theme_light()+
  geom_line(aes(x=evalbind$entrynr, y=evalbind$Agent0_std_punished, colour="Agent LU"),  size = .05, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_line(aes(x=evalbind$entrynr, y=evalbind$Agent1_std_punished, colour="Agent RO"),  size = .05, method="auto", se=FALSE, fullrange=FALSE, level=0.95)+
  geom_smooth(aes(x=evalbind$entrynr, y=evalbind$Agent0_std_punished, colour="Agent LU"),  size = 2, method="auto", se=TRUE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=evalbind$entrynr, y=evalbind$Agent1_std_punished, colour="Agent RO"),  size = 2, method="auto", se=TRUE, fullrange=FALSE, level=0.95)+
  facet_grid(~envnr~epsilon)+
  xlab("Episode")+
  ylab("Unsicherheitsmaß")+
  scale_x_continuous(limits = c(0,7000))+
  guides(colour=guide_legend(title=""))+
  theme(legend.position = 'bottom')

#accuracy eval
ggplot(evalbind)+
  theme_light()+
  facet_wrap(~epsilon)+
  geom_smooth(aes(x=evalbind$entrynr, y=evalbind$correctly_selected, colour = evalbind$nr),  size = 0.25, method="auto", se=FALSE, fullrange=FALSE, level=0.99)+
  geom_smooth(aes(x=evalbind$entrynr, y=evalbind$correctly_selected),  size = 2, method="auto", se=TRUE, fullrange=FALSE, level=0.95)+
  xlab("Episode")+
  ylab("Anteil korrekter Auswahl")+
  scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(limits = c(0,7000))+
  theme(legend.position="none")

