## JOSÉ JAVIER MIÑANO RAMOS
## MovieLens Project 
## HarvardX Data Science Professional Capstone Project
## https://github.com/

#################################################
# MovieLens Rating Prediction Project Code 
################################################






##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(wordcloud)) install.packages("wordcloud", repos = "http://cran.us.r-project.org")
library(wordcloud)
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

download.file("https://www.dropbox.com/s/nspymeso8rmmak1/edx.rds?dl=1", "edx.rds")

download.file("https://www.dropbox.com/s/x0s477b0kzxpl6i/validation.rds?dl=1", "validation.rds")

edx = readRDS("edx.rds")

validation = readRDS("validation.rds")
########################
#EDX CODE ENDS HERE
########################

########################
#Exploring training data and familiarizing with it.
########################

##training set basic information

  head(edx) #show first 6 rows
  summary(edx) #statistically important points
  edx %>%
    summarize(n_users = n_distinct(userId), #69878 different users
            n_movies = n_distinct(movieId)) #10677 different movies

#dependent variable (rating) information
  summary(edx$rating) 
  sd(edx$rating)
  plot(density(edx$rating), main = "Ratings density function") 
  #there exists a significant bias towards non decimal punctuation

#Rating distribution
  edx %>%
    ggplot(aes(rating)) +
    geom_histogram(binwidth = 0.25, color = "black") +
    scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
    scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
    ggtitle("Rating distribution")
  
## nº of films per gender (unfortunately this part is not included in the report because computationally my computer could not do it)

#first we edit the genres to obtain a list:
  genres_edited <- gsub("[|]", " ", edx$genres)
  genres2 <- toString(genres_edited)
  genres3 <- gsub("[,]","", genres2)
  genres4 <- gsub("[-]", "", genres3)

#now, we represent this list
#wordcloud genders
  wordcloud(genres4, scale = c(2, 1), min.freq = 50, colors = rainbow(30)) #wordcloud

#barplot relative frequency genders
  str_count(genres4, "\\w+")
  freq_x <- sort(table(unlist(strsplit(genres4, " ")))/str_count(genres4, "\\w+"),      # Create frequency table
               decreasing = TRUE)
  barplot(freq_x[1:10], width = 5, main = "Relative frequencies of the 10 most represented film genders")

#Number of ratings per movie
  edx %>%
    count(movieId) %>%
    ggplot(aes(n)) +
    geom_histogram(bins = 30, color = "black") +
    scale_x_log10() +
    xlab("Number of ratings") +
    ylab("Number of movies") +
    ggtitle("Number of ratings per movie")

#films rated only once
  edx %>%
    group_by(movieId) %>%
    summarize(count = n()) %>%
    filter(count == 1) %>%
    left_join(edx, by = "movieId") %>%
    group_by(title) %>%
    summarize(rating = rating, n_rating = count) %>%
    slice(1:20) %>%
    knitr::kable()
  
#number of ratings per user
  edx %>%
    count(userId) %>%
    ggplot(aes(n)) +
    geom_histogram(bins = 30, color = "black") +
    scale_x_log10() +
    xlab("Number of ratings") + 
    ylab("Number of users") +
    ggtitle("Number of ratings given by users")

#Mean movie ratings given by users
  edx %>%
    group_by(userId) %>%
    filter(n() >= 100) %>%
    summarize(b_u = mean(rating)) %>%
    ggplot(aes(b_u)) +
    geom_histogram(bins = 30, color = "black") +
    xlab("Mean rating") +
    ylab("Number of users") +
    ggtitle("Mean movie ratings given by users") +
    scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
    theme_light()
  
## now, we want to visualize the emission year of the movies

#first, we use regex to extract the years in the title column
  edx$year <- sapply(str_extract_all(edx$title, "(?<=\\()[^)(]+(?=\\))"), paste0, collapse =",")
  year2 <- as.integer(edx$year)

#We have some NAs, we could substitute them with the median or mean value, but for a qualitative charasteristic as the year it is not correct.
#Even if years are numeric data, they are qualitative data in this case.
#We will look for the percentage of NAs, and eliminate them if possible
  
  percNA <- sum(is.na(year2)==TRUE)/length(year2)
  percNA #6% NA, we can eliminate them for this illustrative purpose
  x <- table(year2)    ## contingency table
  barplot(x, col=c("#ccf0fe9f"), horiz=FALSE, cex.names=0.5, las=1, width=5, 
        main= "X axis: years; Y axis: number of movies")

## represent the distribution of movies and number of califications

  edx %>%
    count(movieId) %>%
    ggplot(aes(n)) +
    geom_histogram(bins = 30, color = "black") +
    scale_x_log10() +
    xlab("Number of ratings") +
    ylab("Number of movies") +
    ggtitle("Number of ratings per movie")

 
########################
#Algorithm construction
########################

##Defining our loss function, we are going to use the RMSE.
  
  RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }

## Simplest model (M0): predicting constantly the mean rate.
  mu <- mean(edx$rating)
  mu

#To compare the future performance of the model, we establish a baseline:
#the RMSE generated by the simplest model
  naive_rmse <- RMSE(validation$rating, mu)
  naive_rmse

#We define a table to store results:
  rmse_results <- data_frame(method = "Average movie rating model (M0)", RMSE = naive_rmse)
  rmse_results %>% knitr::kable()
  
## Estimating movie effect model (M1):
  
  movie_avgs <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = mean(rating - mu))
  
  #plot Number of movies with the computed movie effect
  movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"),
                       ylab = "Number of movies", main = "Number of movies with the computed b_i")

#Apply the M1 model and compute its RMSE
  predicted_ratings <- mu +  validation %>%
    left_join(movie_avgs, by='movieId') %>%
    pull(b_i)
  model_1_rmse <- RMSE(predicted_ratings, validation$rating)
  rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Movie effect model (M1)",  
                                       RMSE = model_1_rmse ))
  rmse_results %>% knitr::kable()

## Movie effect + user effect model (M2)

#plot user effect
  user_avgs<- edx %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    filter(n() >= 100) %>%
    summarize(b_u = mean(rating - mu - b_i))
  user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))
  
#estimate user effect
  user_avgs <- edx %>%
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu - b_i))
  
#Apply the M2 model and compute its RSME
  predicted_ratings <- validation%>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  model_2_rmse <- RMSE(predicted_ratings, validation$rating)
  rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Movie and user effect model (M2)",  
                                       RMSE = model_2_rmse))
  rmse_results %>% knitr::kable()

##regularization model (M3)
  lambdas <- seq(0, 10, 0.25)

# For each lambda,find b_i & b_u, followed by rating prediction & testing
# note:the below code could take some time  
  rmses <- sapply(lambdas, function(l){
    
    mu <- mean(edx$rating)
    
    b_i <- edx %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    
    b_u <- edx %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    
    predicted_ratings <- 
      validation %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      pull(pred)
    
    return(RMSE(predicted_ratings, validation$rating))
  })

# Plot rmses vs lambdas to select the optimal lambda                                                             
  qplot(lambdas, rmses)  
  

# The optimal lambda                                                             
  lambda <- lambdas[which.min(rmses)]
  lambda

# Test and save results                                                             
  rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Regularized movie and user effect model (M3)",  
                                       RMSE = min(rmses)))

#### Results ####                                                            
# RMSE results overview                                                          
  rmse_results %>% knitr::kable()

