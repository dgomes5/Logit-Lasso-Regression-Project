# Daniel Gomes, Blake Simmons, John Gomes
# CIS490 Machine Learning
# catsvsdogs - Logit Regression, Plotting, Variable 93 and 623

# this deletes every variable in the workspce except catsvsdogs and .glm
keep(catsvsdogs, catsvsdogs.glm, sure = TRUE)

# libraries needed
library(caret)
library(graphics)
library(InformationValue) # has optimalCutoff
library(car) # has vif
library(gdata) # has keep 
library(ggplot2)
library(cowplot)
library(pscl)

# load data
if (!exists("catsvsdogs")){
  catsvsdogs <- read.csv("cats-vs-dogs25.csv")
}

training.indices <- sample(1:nrow(catsvsdogs), 0.8 * nrow(catsvsdogs))
training.data <- catsvsdogs[training.indices, ]
testing.data <- catsvsdogs[-training.indices, ]

# these next two lines create our formula for all variables in the set
xnam <- paste0("V", 1:625) 
fmla <- as.formula(paste("V626 ~", paste(xnam, collapse= "+")))

# convert column to categorical
# logistic regression model on whole set
if (!exists("catsvsdogs.glm")){
  catsvsdogs.glm <- glm(fmla, data = catsvsdogs, family = "binomial"(link = "logit"))
}

# gives the beta coefficients, Standard error, z Value and p Value
summary(catsvsdogs.glm)
pR2(catsvsdogs.glm)


# plots some nice graphs that I don't know what they mean yet
plot(catsvsdogs.glm, ask=F)

# convert to factor
catsvsdogs$V626 <- factor(catsvsdogs$V626)

# show predicted probabilites that cat was cat or dog was dog
predicted.data <- data.frame(probability.of.catordog = catsvsdogs.glm$fitted.values, catordog = catsvsdogs$V626)
predicted.data <- predicted.data[order(predicted.data$probability.of.catordog, decreasing = FALSE),]
predicted.data$rank <- 1:nrow(predicted.data)

# plot prediction
ggplot(data = predicted.data, aes(x = rank, y = probability.of.catordog)) +
  geom_point(aes(color = catordog), alpha = 1, shape = 4, stroke = 2) +
  xlab("Index") + ylab("Predicted probability of Cat or Dog") 

# predicted scores
predicted <- predict(catsvsdogs.glm, catsvsdogs, type="response")  

# find the optimal cutoff to improve the prediction of 1’s, 0’s
optCutOff <- optimalCutoff(catsvsdogs$V626, predicted)[1] 

# these need to have VIF well below 4
vif(catsvsdogs.glm)

# The lower the misclassification error, the better is your model.
misClassError(catsvsdogs$V626, predicted, threshold = optCutOff)

# Greater the area under the ROC curve, better the predictive ability of the model.
plotROC(catsvsdogs$V626, predicted)

# higher the concordance, the better is the quality of model
Concordance(catsvsdogs$V626, predicted)

sensitivity(catsvsdogs$V626, predicted, threshold = optCutOff)
specificity(catsvsdogs$V626, predicted, threshold = optCutOff)

# do the confusion matrix
confusionMatrix(catsvsdogs$V626, predicted, threshold = optCutOff)
ctable <- as.table(matrix(c(6788, 4343,5712, 8157), nrow = 2, byrow = TRUE))
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")