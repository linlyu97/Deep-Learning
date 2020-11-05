## 1. Model Architecture
•	Built a ResNet model by adding skip connections as more as possible; used classification train data to train the model

## 2. Training Process

numEpochs = 25
num_feats = 3

## 3. Tricks to Enhance Model Performance

•	Applied center loss as loss criterion and used data augmentation by random horizontal flipping or random vertical flipping the raw pictures in order to improve performance on classification validation data.

•	Calculated cosine similarity through verification data face embedding pairs and used AUC score to evaluate the model performance for face verification.
