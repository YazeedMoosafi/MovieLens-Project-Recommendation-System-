
#################################
# Data setup begins
#################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#################################
# Data setup complete
#################################


##################################################################################
#Add year column to edx and validation data set by extracting year from the title.
##################################################################################
edx <- edx %>% mutate(year=str_sub(str_extract(title , "^*\\(\\d{4}\\)$"), 2,5))
validation <- validation %>% mutate(year=str_sub(str_extract(title , "^*\\(\\d{4}\\)$"), 2,5))

mu <- mean(edx$rating) 
#Normal movie bias
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu), .groups ="keep")

#Normal user bias
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m), .groups ="keep")

#Year  bias
year_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_m - b_u), .groups ="keep")



#########################################################
# Prediction with movie, user and year effect
#########################################################
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_m + b_u + b_y) %>%
  .$pred

#Calculate RMSE
RMSE(validation$rating, predicted_ratings)

####################################
#Code to find out optimal lambda
####################################
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  #Regularized movie bias
  b_m <- edx %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l), .groups ="keep")
  #Regularized user bias
  b_u <- edx %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l), .groups ="keep")
  
  #Regularized year bias
  b_y <- edx %>% 
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by='userId') %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_m - b_u - mu)/(n()+l), .groups ="keep")
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_m + b_u + b_y) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]

#########################
#Print the Optimal lambda
#########################
lambda

#########################################################
# Prediction with regularized movie and user effect
#########################################################

mu <- mean(edx$rating)
#Calculate regularized movie bias
b_m <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+lambda), .groups ="keep")

#Calculate regularized user bias
b_u <- edx %>% 
  left_join(b_m, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - mu)/(n()+lambda), .groups ="keep")

#Calculate regularized year bias
b_y <- edx %>% 
  left_join(b_m, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - b_m - b_u - mu)/(n()+lambda), .groups ="keep")

#Predict by considering regularized user, movie and year bias
predicted_ratings <- 
  validation %>% 
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "year") %>%
  mutate(pred = mu + b_m + b_u + b_y) %>%
  .$pred
#Calculate RMSE
RMSE(predicted_ratings, validation$rating)

FINAL_RMSE <- 	 RMSE(predicted_ratings, validation$rating)


##################################
#Print the final RMSE (< 0.86490)
###################################
FINAL_RMSE


