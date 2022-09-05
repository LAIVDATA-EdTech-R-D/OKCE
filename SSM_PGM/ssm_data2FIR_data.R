library(reshape2)


make_reshaped_table  <- function(dat){
  dat1=as.data.frame(acast(dat,knowre_user_id ~ kc_uid,value.var="accuracy", fun=mean))
  return(dat1)
}

input_file_name = "../data/kc_dedup_smath12.csv"

analysis_data <- read.csv(input_file_name) # input file should be a matrix
analysis_dat <- make_reshaped_table(analysis_data[substr(analysis_data$test,15,15)=='1',]) # make a horizontally wide table "knowre_user_id~ kc_uid in terms of test1"

analysis_data0 <- analysis_dat[rowSums(analysis_dat[])<ncol(analysis_dat[]) & rowSums(analysis_dat[]) > 0 ,]    
analysis_data1 <- analysis_data0[,colSums(analysis_data0[]) !=nrow(analysis_data0[])]

write.csv(analysis_data1,"../data/kc_dedup_smath12_reshape.csv")
