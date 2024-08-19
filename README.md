# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
First we can take the dataset based on one input value and some mathematical calculus output value.Next define the neural network model in three layers.First layer has six neurons and second layer has four neurons,third layer has one neuron.The neural network model takes the input and produces the actual output using regression.

## Neural Network Model
![Screenshot 2024-08-19 143556](https://github.com/user-attachments/assets/c306851c-f162-4c80-bfd9-5148fd030c7e)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Adhithya M R
### Register Number: 212222240002
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab  import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds,_=default()
gc = gspread.authorize(creds)
worksheet =  gc.open('data').sheet1
data = worksheet.get_all_values()
df=pd.DataFrame(data[1:], columns=data[0])
df=df.astype({'X':'int'})
df=df.astype({'Y':'int'})
df.head()
X = df[['X']].values
Y = df[['Y']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer= 'rmsprop',loss = 'mse')
ai_brain.fit(X_train1,Y_train,epochs= 2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,Y_test)
X_n1 = [[6]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1 )
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,Y_test)
X_n1 = [[6]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1 )

```
## Dataset Information
![Screenshot 2024-08-19 143309](https://github.com/user-attachments/assets/322e91f7-482f-4535-a86c-bda40b1337e0)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-08-19 144106](https://github.com/user-attachments/assets/0f4bb2a3-ab81-45f1-a189-ea7da603523e)

### Test Data Root Mean Squared Error
![Screenshot 2024-08-19 144250](https://github.com/user-attachments/assets/01f67a74-78d4-45d2-8209-238e895162d8)


### New Sample Data Prediction

![Screenshot 2024-08-19 144408](https://github.com/user-attachments/assets/1fb225bb-f924-40a6-b192-0b7da8505b7a)


## RESULT
Thus a Neural network for Regression model is Implemented.


