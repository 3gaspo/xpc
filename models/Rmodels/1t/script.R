rm(list = ls())
suppressPackageStartupMessages(require(R39Toolbox))
library(readr)

# Load =  factor(Instant) + factor(DayType) + s(temp)


# chemin fichier pour lire les données dont on souhaite faire la prévision
arg1 = 'models/Rmodels/1t/input.csv'

# chemin fichier pour écrire le prévision
arg2 = 'models/Rmodels/1t/output.csv'

data <- suppressMessages(read_delim(arg1, delim = ',')) %>% 
  mutate(Instant = as.factor(Instant),
         DayType = as.factor(DayType))

model <- readRDS("models/Rmodels/1t/model.rds")

prev <- predict_details(model, data) 

col_PC <- c("s(temp)")
column_list <- c("as.factor(Instant)","as.factor(DayType)","s(temp)")

for (col in column_list){
  pred_col = paste0("pred_",col)
  data[[pred_col]] <- prev[,col]
}

data$EstimatedLoad <- rowSums(prev)
data$EstimatedLoad_PC <-  sum(prev[,col_PC])
data$EstimatedLoad_PHC <- data$EstimatedLoad  - data$EstimatedLoad_PC 

write_csv(data, arg2)