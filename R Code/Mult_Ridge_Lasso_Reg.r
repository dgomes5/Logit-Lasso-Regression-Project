# Blake Simmons, Daniel Gomes, John Gomes
# CIS490
# Project 1

library(ridge)
library(glmnet)
library(ggplot2)
library(jtools)


#import, naming, and partitioning of data
housing <- read.table('housing.data', header = FALSE)
names(housing) <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")
housing$CHAS = as.logical(housing$CHAS)
sapply(housing, class)
set.seed(86)
training.indices <- sample(1:nrow(housing), 0.8 * nrow(housing))


#multiple linear regression
training.data <- housing[training.indices, ]
testing.data <- housing[-training.indices, ]

frml <- MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT
multi.var.model <- lm(frml, data = training.data)
print(summary(multi.var.model))

multi.var.predictions <- predict(multi.var.model, testing.data)
test.multi.var.ssl <- sum((testing.data$CRIM - multi.var.predictions)^2)
sprintf("SSL/SSR/SSE: %f", test.multi.var.ssl)
test.multi.var.mse <- test.multi.var.ssl / nrow(testing.data)
sprintf("MSE: %f", test.multi.var.mse)
test.multi.var.rmse <- sqrt(test.multi.var.mse)
sprintf("RMSE: %f", test.multi.var.rmse)

#effect_plot(multi.var.predictions, pred = MEDV, interval = TRUE, plot.points = TRUE)

full.model <- lm(frml, data = housing)
print(summary(full.model))

#testing data plots
scatter.smooth(x = testing.data$ZN, y = testing.data$MEDV, main="MEDV ~ ZN")
scatter.smooth(x = testing.data$INDUS, y = testing.data$MEDV, main="MEDV ~ INDUS")
scatter.smooth(x = testing.data$CHAS, y = testing.data$MEDV, main="MEDV ~ CHAS")
scatter.smooth(x = testing.data$NOX, y = testing.data$MEDV, main="MEDV ~ NOX")
scatter.smooth(x = testing.data$RM, y = testing.data$MEDV, main="MEDV ~ RM")
scatter.smooth(x = testing.data$AGE, y = testing.data$MEDV, main="MEDV ~ AGE")
scatter.smooth(x = testing.data$DIS, y = testing.data$MEDV, main="MEDV ~ DIS")
scatter.smooth(x = testing.data$RAD, y = testing.data$MEDV, main="MEDV ~ RAD")
scatter.smooth(x = testing.data$TAX, y = testing.data$MEDV, main="MEDV ~ TAX")
scatter.smooth(x = testing.data$PTRATIO, y = testing.data$MEDV, main="MEDV ~ PTRATIO")
scatter.smooth(x = testing.data$B, y = testing.data$MEDV, main="MEDV ~ B")
scatter.smooth(x = testing.data$LSTAT, y = testing.data$MEDV, main="MEDV ~ LSTAT")
scatter.smooth(x = testing.data$CRIM, y = testing.data$MEDV, main="MEDV ~ CRIM")



#ridge regression
housing.mat <- model.matrix(MEDV ~ ., data = housing)
housing.mat <- housing.mat[,-13]

X <- housing.mat
Y <- housing[,'MEDV']



X.train <- X[training.indices,]
X.test <- X[-training.indices,]
Y.train <- Y[training.indices]
Y.test <- Y[-training.indices]

linear.mod <- lm(Y.train ~ X.train)
#summary(linear.mod)

lm.pred <- predict(linear.mod, newx = X.test)
rmse <- sqrt(mean((lm.pred - Y.test)^2))
sprintf("RMSE: %f", rmse)

grid <- 10^seq(10, -2, length=1000)
ridge.mod <- glmnet(X.train, Y.train, alpha=0, lambda=grid, thresh=1e-12)
cv.out <- cv.glmnet(X.train, Y.train, alpha=0, nfolds = 10)
plot(cv.out)

best.lambda <- cv.out$lambda.min
sprintf("Ridge Best Lambda: %f", best.lambda)

ridge.pred <- predict(ridge.mod, s=best.lambda, newx = X.test)
print(paste('RMSE:', sqrt(mean((ridge.pred - Y.test)^2))))
#ridge.pred

ridge.out <- glmnet(X, Y, alpha = 0)
plot(ridge.out, xvar = "lambda", ylim = c(-10, 10))

ridge.results <- predict(ridge.out, type = "coefficients", s=best.lambda)
ridge.results

#LASSO Regression model
lasso.mod <- glmnet(X.train, Y.train)
plot(lasso.mod, xvar="lambda")

lasso.cv.out <- cv.glmnet(X, Y, alpha=1, nfolds = 10)
plot(lasso.cv.out)

lasso.mod.pred <- predict(lasso.mod, X.test)
rmse <- sqrt(apply((lasso.mod.pred - Y.test)^2, 2, mean))
lasso.best.lambda <- lasso.cv.out$lambda.1se
print(lasso.best.lambda)

lasso.pred <- predict(lasso.mod, s=lasso.best.lambda, newx=X.test)
print(paste('RMSE:', sqrt(mean((lasso.pred - Y.test)^2))))

lasso.best.lambda

lasso.out <- glmnet(X, Y, alpha=1, lambda=grid)
lasso.coef <- predict(lasso.out, type="coefficients", s=lasso.best.lambda)
lasso.coef


