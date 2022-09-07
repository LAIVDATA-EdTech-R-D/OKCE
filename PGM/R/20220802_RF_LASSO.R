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
  info <- fromJSON(file = "./PGM/info_back.json")
  
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
    glmnet1<-cv.glmnet(x=x,y=dat[,target_name],type.measure='mse',nfolds=as.integer(nrow(dat)/3) ,alpha=1)
    c<-coef(glmnet1,s='lambda.min',exact=TRUE)
    inds<-which(c!=0)
    variables<-row.names(c)[inds]
    variables<-variables[variables %ni% c('(Intercept)')]
    if(length(variables)==0) next
    dat_temp = dat[,c( target_name,variables)]
    
    formula1 = paste(target_name ,"~.")
    
    print(formula1)
#   print(summary(lm(formula=formula1, dat_temp)))
    
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
#2. Run Random Forest
#-------------------------------------------------------------------------------------------------------------#

run_rf <- function(dat){
  
  imp_mat <- data.frame(before=character(), after=integer(),character(), imp = double())

  for(i in 1:ncol(dat)){
    print(paste0("Calculate importance values on ",colnames(dat)[i]))
    set.seed=500
    imp <-  importance(randomForest(dat[,-i], dat[,i],ntree=500, importance=T, scale=TRUE),type=1)
    imp_temp <- as.data.frame(cbind(rownames(imp), matrix(rep(colnames(dat)[i],nrow(imp)), nrow= nrow(imp), ncol=1,byrow=TRUE), imp[,1]))
    colnames(imp_temp) = c("before","after","imp")
    imp_mat = rbind(imp_mat,imp_temp)
  }
  imp_mat$imp=as.integer(imp_mat$imp)  
  print("Congratulations : Importance matrix was made !!!")
  return(imp_mat)
}


#-------------------------------------------------------------------------------------------------------------#
#3. source from BOTH, LASSO, RF
#-------------------------------------------------------------------------------------------------------------#

check_from_merge <- function(lasso, rf){

  lasso$from1 = "LASSO"
  rf$from2 = "RF"
  
  lasso_rf= merge(x=rf,y=lasso, by=c("before","after","imp"), all=TRUE)
  print(paste0("RF ", nrow(rf)," rows and LASSO  ", nrow(lasso), " rows were merged to lasso_rf ", nrow(lasso_rf)," rows"))
  
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
#5. make 12 lines from each KC 
#-------------------------------------------------------------------------------------------------------------#

make_final_set <- function(rf_lasso_dat, rf_imp_dat, col_names){

  final_set = merge(x=rf_lasso_dat, y=rf_imp_dat, by=c("before","after"),all.x=TRUE)
  final_set$ord= ifelse(final_set$method=='BOTH', 1 ,ifelse(final_set$method=='LASSO',2,3))
  colnames(final_set)=c("before","after","impx","method","imp","ord")
  fset= sqldf("SELECT * FROM (SELECT before, after, ord, method,imp, ROW_NUMBER() 
                              OVER(PARTITION BY before ORDER BY before, ord, imp DESC) as rn
          FROM final_set) WHERE rn <= 12")
  
  
  #make candidate rows to add to final set
  
  #1. select "before"s when addition is needed.
  
  temp1 = sqldf("select before, max(rn) as max_rn from fset group by before")
  temp2 = temp1[temp1$max_rn < 12,]
  temp2$add_rn = 12-temp2$max_rn
  
  #2. make possible pair for lacking "before"s .
  
  temp3 = data.frame(col= col_names)
  temp4 = sqldf("select before, col as after, add_rn from temp2, temp3") 
  
  # bring all columns for the 2. "before's
  
  temp5 = merge(fset, temp2, by="before", all.y=TRUE)
  temp5$exists = 1
  
  # check the obs is if appeared or not among all obs
  temp6 = merge(temp5, temp4, by=c("before","after"), all.y=TRUE)[,c("before","after","exists","add_rn.y")]
  colnames(temp6)[4]="add_rn"
  
  # candidates for random sampling
  candidates = temp6[is.na(temp6$exists),]                                                                     
  
  
  temp7 = sqldf("select *, random() as rand from candidates")
  
  temp8 = sqldf("select * from (select before, after, add_rn, ROW_NUMBER() OVER(PARTITION BY before  ORDER BY rand) as rank from temp7) where rank <= add_rn ")
  
  # bind final set with random samples 
  fset_add = cbind(temp8$before, temp8$after, NA,NA,NA,NA)
  colnames(fset_add)= colnames(fset)
  fset1=rbind(fset,fset_add)
  
  return(fset1)

}

#-------------------------------------------------------------------------------------------------------------#
#6. make data to transfer to selector net/operator/net
#-------------------------------------------------------------------------------------------------------------#


run <- function(){

    tic("read tables for relation analysis")    
    analysis_data <- read.csv(input_file_name)[,-1] # input file should be a matrix
#   analysis_data <- read.csv("/home/laivdata/바탕화면/choihh/FIR/DATA/algebra_picture3000.csv")[,-1] # input file should be a matrix
    analysis_data1 <- analysis_data[rowSums(analysis_data[])<ncol(analysis_data[]) & rowSums(analysis_data[]) > 0 ,]    
    summary(analysis_data1)[4,] #mean for all columns
    toc()
  
#----------------------------------Elastic LASSO + random forest ---------------------------------------  
    tic("Discover relationships b/w KCs based on elastic net results (CV=1000, lasso)")
    rel_elasso <- unique(elastic_anal(analysis_data1))
    toc()
  
    tic("Relationships were found from random forest algorithm")
    rf_imp <- run_rf(analysis_data1)
    toc()     

    # since rf scores all of the relations nC2, rel-elasso can be a trimmed result through merging.
    rel_elasso1 = merge(x=rf_imp, y=rel_elasso, by=c("before","after"), all.y=TRUE)
    
    tic("Select top relations among all possible relations")    
    if(rf_cut < 0.3){ # 0.1 : top 10% , 0.2 : top 20% or larger than 0.3 : same length with elastic net
      rf_df1 <- sqldf(paste0("select * from rf_imp order by imp desc limit ",as.integer(nrow(rf_imp)*rf_cut)))
      print(paste0("The top ", rf_cut*100, "% records of the RF result were extracted"))
    } else {
      rf_df1 <- sqldf(paste0("select * from rf_imp order by imp desc limit ",as.integer(nrow(rel_elasso))))  
    }  
    toc()
#----------------------------------Merge RF & LASSO ---------------------------------------  
    
    tic("Merge LASSO and RF checking BOTH, LASSO, RF")
    rf_lasso=check_from_merge(rel_elasso1, rf_df1)[,c("before","after","imp","method")]
    toc()

    tic("Make a final set according to the order BOTH, LASSO, RF")    
    relation_f <- make_final_set(rf_lasso,rf_imp, colnames(analysis_data1))
    relation_f1 = relation_f[order(relation_f$before),]    
    toc()
    write.csv(relation_f1,"./OUT/relation.csv")
    
    
}

init()

run()

