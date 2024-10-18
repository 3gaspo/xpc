rm(list = ls())
suppressPackageStartupMessages(require(R39Toolbox))
library(readr)

# Load = offset + Instant + DayType + s(temp) + s(temp_liss_fort) + s(temp_liss_faible) + s(Posan) + s(tempMax) + s(wind) + s(sun) + s(tempMin)


# chemin fichier pour lire les données dont on souhaite faire la prévision
arg1 = 'models/Rmodels/1/input.csv'

# chemin fichier pour écrire le prévision
arg2 = 'models/Rmodels/1/output.csv'

data <- suppressMessages(read_delim(arg1, delim = ',')) %>% 
  mutate(offset = as.factor(offset),
         Instant = as.factor(Instant),
         DayType = as.factor(DayType))

model <- readRDS("models/Rmodels/1/model.rds")

prev <- predict_details(model, data) 

col_PC <- c("s(temp)", "s(temp_liss_fort)", "s(temp_liss_faible)", 
            "s(tempMax)", "s(tempMin)",  "s(wind)" ,"s(sun)")


column_list <- c("offset","as.factor(Instant)","DayType","s(temp)","s(temp_liss_fort)","s(temp_liss_faible)","s(Posan)","s(tempMax)","s(wind)","s(sun)","s(tempMin)")

for (col in column_list){
  #print(col)
  pred_col = paste0("pred_",col)
  data[[pred_col]] <- prev[,col]
}


if (nrow(prev)>1) {
  data$EstimatedLoad <- rowSums(prev)
  data$EstimatedLoad_PC <-  rowSums(prev[,col_PC])
} else {
  data$EstimatedLoad <- rowSums(prev)
  data$EstimatedLoad_PC <-  sum(prev[,col_PC])
}

data$EstimatedLoad_PHC <- data$EstimatedLoad  - data$EstimatedLoad_PC 

write_csv(data, arg2)