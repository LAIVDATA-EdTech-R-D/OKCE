##################################################################################################
# The purpose of this program is finding the relationships of 2KCs w/ lasso+rf
##################################################################################################

library(rjson)
library(aws.s3)
library(tictoc)
library(sqldf)
library(reshape2)
library(glmnet)
library(stringr)
library(purrr)
library(plyr)
library(caret)
library(randomForest)
library(plyr)


init <- function() {
  info <- fromJSON(file = "./info_back.json")
  
  # host parameters
  useaws <<- info[[1]]$useaws
  bucketname <<- info[[1]]$bucketname
  localpath <<- info[[1]]$localpath

  # csv file name
  input_file_name <<- info[[1]]$input_file_name
  lasso_p_value <<-info[[1]]$lasso_p_value
  rf_cut <<-info[[1]]$rf_cut
  print (useaws)
  print (bucketname)
  print (localpath)
  print (input_file_name)
  print (lasso_p_value)
  print (rf_cut)
  lasso_p_value = 0.1
  rf_cut = 0.2
}  



#-------------------------------------------------------------------------------------------------------------#
#1. elastic net
#-------------------------------------------------------------------------------------------------------------#

elastic_anal <- function(dat){

  kcs=colnames(dat)

  lasso_relation <- data.frame(kc1=character(), kc2=character())  #Define a table to save relations b/w KC
  
  for(i in 1:length(kcs)){
    target_name = kcs[i]
    `%ni%`<-Negate("%in%")
    x<-model.matrix(dat[,target_name]~.,data=dat)
    x=subset(x, select= -c(which(startsWith(colnames(x), target_name))))
    set.seed(100) 
    glmnet1<-cv.glmnet(x=x,y=dat[,target_name],type.measure='mse',nfolds=1000,alpha=1)
    c<-coef(glmnet1,s='lambda.min',exact=TRUE)
    inds<-which(c!=0)
    variables<-row.names(c)[inds]
    variables<-variables[variables %ni% c('(Intercept)')]
    if(length(variables)==0) next
    dat_temp = dat[,c( target_name,variables)]
    
    formula1 = paste(target_name ,"~.")
    
    print(formula1)
    print(summary(lm(formula=formula1, dat_temp)))
    
    prob = as.matrix(summary(lm(formula=formula1, dat_temp))$coefficients)
    
    prob1 = row.names(prob[prob[,4]<= lasso_p_value,])  
    prob2 = prob1[prob1 %ni% '(Intercept)']
    if(length(prob2)==0) next
    
    for( j in 1:length(prob2)) lasso_relation = rbind(lasso_relation, cbind(prob2[j],target_name))
    
  }
  colnames(lasso_relation)=c("before","after")
  
  rel = sqldf("select before , after from lasso_relation  order by before, after")

  return(rel)
}  

#-------------------------------------------------------------------------------------------------------------#
#2. Run random forest
#-------------------------------------------------------------------------------------------------------------#

run_rf <- function(dat){
  
#  dev.set(which = dev.next())

  imp_mat <- data.frame("before" = colnames(dat)[1], "after" = colnames(dat)[2], imp = 300)
  
  for(i in 3:ncol(dat)){
    set.seed=500
    imp <-  importance(randomForest(dat[,1:i-1], dat[,i],ntree=500, importance=T, scale=TRUE),type=1)
    imp_temp <- as.data.frame(cbind(rownames(imp), matrix(rep(colnames(dat)[i],nrow(imp)), nrow= nrow(imp), ncol=1,byrow=TRUE), imp[,1]))
    colnames(imp_temp) = c("before","after","imp")
    imp_mat = rbind(imp_mat,imp_temp)
  }
  
  imp_mat$imp=as.integer(imp_mat$imp)
  
#  dev.off(which = dev.cur())
  return(imp_mat)
}


#-------------------------------------------------------------------------------------------------------------#
#3. source from BOTH, LASSO, RF
#-------------------------------------------------------------------------------------------------------------#

check_from_merge <- function(lasso, rf){


  lasso$from1 = "LASSO"
  rf$from2 = "RF"
  
  lasso_rf= merge(x=rf,y=lasso, by=c("before","after"), all=TRUE)
  
  lasso_rf$method=ifelse(is.na(lasso_rf$from1) & is.na(lasso_rf$from2),"0",
                        ifelse(is.na(lasso_rf$from1),"RF",
                               ifelse(is.na(lasso_rf$from2),"LASSO",
                                      "BOTH")))
  lasso_rf$from1=NULL
  lasso_rf$from2=NULL
  
  return(lasso_rf)
  
}

#-------------------------------------------------------------------------------------------------------------#
#4. Since there is no order, birelation is not meaningful
#-------------------------------------------------------------------------------------------------------------#

clean_birel <- function(dat){
  dat2 = sqldf("select min(before, after) as before, max(before,after) as after,max(imp) as imp from dat group by before, after order by before, after")
  return(dat2)
}

#-------------------------------------------------------------------------------------------------------------#
#5. make 15 lines from each KC 
#-------------------------------------------------------------------------------------------------------------#

make_final_set <- function(rf_lasso_dat, rf_imp_dat, col_names){
  rf_lasso_dat=rf_lasso
  rf_imp_dat = rf_imp
  final_set = merge(x=rf_lasso_dat, y=rf_imp_dat, by=c("before","after"),all.x=TRUE)
  final_set$ord= ifelse(final_set$method=='BOTH', 1 ,ifelse(final_set$method=='LASSO',2,ifelse(final_set$method=='RF', 3,4)))
  final_set[is.na(final_set$ord),c("ord")]=4
  final_set[is.na(final_set$method),c("method")]="_"
  colnames(final_set)=c("before","after","impx","method","imp","ord")
  fset = sqldf("select before, after, ord, method, imp from final_set order by before, ord, imp desc")
  
  fset= sqldf("SELECT * FROM (SELECT before, after, ord, method,imp, ROW_NUMBER() 
                              OVER(PARTITION BY before ORDER BY before, ord, imp DESC) as rn
          FROM final_set) WHERE rn <= 12")

  # select "before"s when addition is needed.
  
  temp1 = sqldf("select before, max(rn) as max_rn from fset group by before")
  temp2 = temp1[temp1$max_rn < 12,]
  temp2$add_rn = 12-temp2$max_rn
  
  # make possible pair only for "before"s which are requesting addition of rows.
  
  temp3 = data.frame(col= (colnames(analysis_data1)))
  temp4 = sqldf("select before, col as after, add_rn from temp2, temp3") 
  
  # only select observations of target "before"s
  
  temp5 = merge(fset, temp2, by="before", all.y=TRUE)
  temp5$exists = 1
  
  fset2 = sqldf("select before, 10-max_rn as cnt from fset1 where max_rn <10")

  fset_add1 = merge(x=fset, y=fset2, by="before", all.y=TRUE)
  fset_add2 = cbind(sqldf("select before, after from fset_add1 where rn <= cnt  "),NA,NA,NA,NA)  
  colnames(fset_add2)=colnames(fset)
  fset3 = rbind(fset, fset_add2)
  return(fset)
}

#-------------------------------------------------------------------------------------------------------------#
#6. make data to transfer to selector net/operator/net
#-------------------------------------------------------------------------------------------------------------#

random_select <- function(dat){
  for(i in 1:10){
    library(dplyr)
    out <- dat %>% group_by(dat$before) %>% slice_sample(n=10)
    write.csv(out, paste0("./OUT/out",i,".csv"))     
  }  
}  

init()


    tic("make series table and markov table with 2 test results w/o backing information")    
#    analysis_data <- read.csv(input_file_name) # input file should be a matrix
    analysis_data <- read.csv("/home/laivdata/바탕화면/FIR/MERGE/data/algebra_percent3000.csv")[,-1] # input file should be a matrix
    analysis_data1 <- analysis_data[rowSums(analysis_data[])<ncol(analysis_data[]) & rowSums(analysis_data[]) > 0 ,]    
    summary(analysis_data1)
    toc()
  
#----------------------------------Elastic LASSO + random forest ---------------------------------------  
    tic("Discover relationships b/w KCs based on elastic net results (CV=1000, lasso)")
    rel_elasso <- elastic_anal(analysis_data1)
    toc()
  
    tic("run random forest algorithm")
    rf_imp <- run_rf(analysis_data1)
    rf_df <- clean_birel(rf_imp)    
  
    if(rf_cut < 0.3){
      rf_df1 <- sqldf(paste0("select * from rf_df order by imp desc limit ",as.integer(nrow(rf_imp)*rf_cut)))
    } else {
      rf_df1 <- sqldf(paste0("select * from rf_df order by imp desc limit ",as.integer(nrow(rel_elasso))))  
    }  
    toc()
#----------------------------------Merge RF & LASSO ---------------------------------------  
    
    tic("merge LASSO and RF")
    rf_lasso <- check_from_merge(rel_elasso, rf_df1)
    toc()

#----------------------------------Make final set  with 15 lines---------------------------------------          
    relation_f <- make_final_set(rf_lasso,rf_imp, colnames(analysis_data1))
    sqldf("select before, max(rn) from relation_f group by  before")

    
#----------------------------------random select with 10 iterations---------------------------------------              
    random_select(relation_f)
    
