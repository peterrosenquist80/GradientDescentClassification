
# This script is a short exercise from a machine learning course. It 
# implements a common machine learning algorithm known as gradient descent
# to predict students' likelihood of admission based on test scores and
# admission of past students.

# !-- Need to set working directory to location of data.txt file
Data <- read.csv('data.txt', header = F)
# create function for plotting scores and indicating admission
plotData <- function(X, Y){
  pos <- which(Y == 1)
  neg <- which(Y == 0)
  plot(X[pos, 1], X[pos, 2], 
       xlim = c(30, 100),
       ylim = c(30, 100),
       pch = 3, 
       col = 'blue',
       xlab = 'Exam 1 Score', 
       ylab = 'Exam 2 Score',
       title("Students Admitted and Exam Scores")
       )
  points(X[neg, 1], X[neg, 2], 
         pch = 1,
         col = 'red'
         )
  legend(x = 80, y = 100, 
         legend = c('admitted', 'not admitted'),
         pch = c(3, 1),
         col = c('blue', 'red'),
         cex = .9
         )
}
# plot with dataset
plotData(Data[,1:2], Data[,3])
readline(prompt="Press [enter] to continue")
# sigmoid function for logistic regression evaluation
sigmoid <- function(z){
  g <- 1 / (1 + exp(z))
  return(g)
}
# create cost function for use in gradient descent
costFunction <- function(theta, X, Y){
  m <- length(Y)
  h_x <- X %*% theta
  J_insides <- -Y * log(sigmoid(h_x)) - (1 - Y) * log(1 - sigmoid(h_x))
  J <- 1 / m * sum(J_insides)
  return(J)
}
# compute gradient 
gradientFunction <- function(X, Y){
  grad_insides <- sweep(X, MARGIN=1, (sigmoid(h_x) - Y), "*")
  grad <- 1 / m * colSums(grad_insides)
  return(grad)
}
# convert to matrix form, necessary for optim function
# also add column of 1's to account for theta_0 x values
Data <- as.matrix(cbind(rep(1, 100), Data))
colnames(Data) <- c('X0', 'X1', 'X2', 'Y')
# runs gradient descent, with costFunction as the cost function 
grad_descent <- optim(par = c(0, 0, 0), fn = costFunction, X = Data[,1:3], Y = Data[,4])
print(grad_descent$value)
# save optimal theta values
optim_theta <- grad_descent$par

# create coordinates for boundary line
x_coordinates = c(min(Data[,2])-2,  max(Data[,2])+2)
y_coordinates = (-1/optim_theta[3]) * (optim_theta[2] * x_coordinates + optim_theta[1])
# plot original graph with added boundary line 
plotData(Data[,2:3], Data[,4])  
lines(x = x_coordinates, 
      y = y_coordinates
     ) 
# update legend to include boundary
legend(x = 80, y = 100, 
       legend = c('admitted', 'not admitted', 'decision boundary'),
       pch = c(3, 1, NA), lty = c(NA, NA, 1),
       col = c('blue', 'red', 'black'),
       cex = .9
      )
# a few examples of hypothetical students' likelihood of admission
# student with Test Score 1: 70, Test Score 2: 55
sigmoid( c(1, 70, 55) %*% optim_theta)
# probability of admission: .588
# student with Test Score 1: 90, Test Score 2: 85
sigmoid( c(1, 90, 85) %*% optim_theta)
# probability of admission: .999
# student with Test Score 1: 60, Test Score 2: 45
sigmoid( c(1, 60, 45) %*% optim_theta)
# probability of admission: .023