
library(data.table)

print("Reading data")
train <- fread('../input/train.csv', 
               select = c('Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Demanda_uni_equil'))

test <- fread('../input/test.csv', 
              select = c('id', 'Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK'))

print("Computing means")
#transform target variable to log(1 + demand) - this makes sense since we're 
#trying to minimize rmsle and the mean minimizes rmse:
train$log_demand = 1.006*log1p(train$Demanda_uni_equil+0.01)-0.006
mean_total <- mean(train$log_demand) #overall mean

#mean by product
mean_P <-  train[, .(MP = mean(log_demand)), by = .(Producto_ID)]
#mean by ruta
mean_C <-  train[, .(MC = mean(log_demand)), by = .(Cliente_ID)]
#mean by product and agencia
mean_PA <- train[, .(MPA = mean(log_demand)), by = .(Producto_ID, Agencia_ID)] 
#mean by product and ruta
mean_PR <- train[, .(MPR = mean(log_demand)), by = .(Producto_ID, Ruta_SAK)] 
#mean by product, client, agencia
mean_PCA <- train[, .(MPCA = mean(log_demand)), by = .(Producto_ID, Cliente_ID, Agencia_ID)]

print("Merging means with test set")
submit <- merge(test, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
submit <- merge(submit, mean_PA, all.x = TRUE, by = c("Producto_ID", "Agencia_ID"))
submit <- merge(submit, mean_C, all.x = TRUE, by = "Cliente_ID")
submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")

# Now create Predictions column;
submit$Pred <- expm1(submit$MPCA)*0.72+expm1(submit$MPR)*0.189+0.1
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MPR)*0.746+0.17
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MC)*0.82+0.86
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MPA)*0.54+0.8
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MP)*0.44+1
submit[is.na(Pred)]$Pred <- expm1(mean_total)

submit$Pred <- round(submit$Pred,5)

print("Write out submission file")
# now relabel columns ready for creatig submission
setnames(submit,"Pred","Demanda_uni_equil")
# Any results you write to the current directory are saved as output.
write.csv(submit[,.(id,Demanda_uni_equil)],"submit_mean_by_Agency_Ruta_Client.csv", row.names = FALSE)
print("Done!")
