import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import random

df = pd.read_csv('Telco-Customer-Churn.csv')

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Convert categorical columns to numerical using LabelEncoder
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Scale numerical columns
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Convert target column to numerical
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y

X, y = preprocess_data(df)

#Balancing the Data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

#80:20 Split
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#Function to set list of weights to the neural network model
def set_weights(model, weights):
    flattened_weights = np.array(weights)
    shapes = [w.shape for w in model.get_weights()]
    index = 0
    new_weights = []
    for shape in shapes:
        size = np.prod(shape)
        layer_weights = flattened_weights[index:index+size].reshape(shape)
        index += size
        new_weights.append(layer_weights)
    model.set_weights(new_weights)
    return model

#Fitness Function
def fitness_func(model, weights):
    model = set_weights(model, weights)
    return model.evaluate(x_train, y_train, verbose = 1)[1]

#Basic Version of Firefly Algorithm
def firefly_algo(model, fitness, num_fireflies, alpha, beta, gamma, max_limit):
    fireflies=[]
    weights = model.get_weights()
    weights_vector = np.concatenate([w.flatten() for w in weights])
    for i in range(num_fireflies):
        temp = weights_vector.copy()
        rand_vec = np.random.uniform(2,-2,len(weights_vector))
        temp*=rand_vec
        fireflies.append(temp)
    fireflies = np.array(fireflies)
    for _ in range(max_limit):
        fitnesses = np.array([fitness(model,firefly) for firefly in fireflies])
        indices = np.argsort(-fitnesses)
        fireflies = fireflies[indices]
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                r = np.linalg.norm(fireflies[i] - fireflies[j])
                beta_ = beta*np.exp(-gamma*r**2)
                fireflies[i] = fireflies[i] + beta_ * (fireflies[j] - fireflies[i]) + alpha * np.random.normal(size=fireflies[j].shape)     
    return set_weights(model, fireflies[0])

#New Version of Firefly Algorithm
def firefly_algo2(model, fitness, num_fireflies, alpha, beta, gamma, max_limit):
    # Initialize fireflies randomly
    fireflies=[]
    weights = model.get_weights()
    weights_vector = np.concatenate([w.flatten() for w in weights])
    for i in range(num_fireflies):
        temp = weights_vector.copy()
        rand_vec = np.random.uniform(2,-2,len(weights_vector))
        temp*=rand_vec
        fireflies.append(temp)
    fireflies = np.array(fireflies)
    fitnesses = np.array([fitness(model,firefly) for firefly in fireflies])
    # Main loop of the algorithm
    for _ in range(max_limit):
        # Update each firefly based on its attractiveness to others
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if fitnesses[j] > fitnesses[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta_ = beta * np.exp(-gamma * r ** 2)
                    fireflies[i] = fireflies[i] + beta_ * (fireflies[j] - fireflies[i]) + alpha * np.random.normal(size=fireflies[j].shape)

            # Update fitness of the firefly
            fitnesses[i] = fitness(model, fireflies[i])

    # Select the best firefly as the solution
    best_idx = np.argmax(fitnesses)
    best_firefly = fireflies[best_idx]

    return set_weights(model, best_firefly)

#Neural Network Model Definition
def baseline_model():
    input_shape = [x_train.shape[1]]
    model = keras.Sequential()
    initializer = keras.initializers.GlorotUniform()
    model.add(keras.layers.Dense(38, kernel_initializer=initializer, input_shape=input_shape, activation='sigmoid'))
    model.add(keras.layers.Dense(76, kernel_initializer=initializer, activation='sigmoid'))
    model.add(keras.layers.Dense(76, kernel_initializer=initializer, activation='sigmoid'))
    model.add(keras.layers.Dense(38, kernel_initializer=initializer, activation='sigmoid'))
    model.add(keras.layers.Dense(1, kernel_initializer=initializer, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


model = baseline_model()
history = firefly_algo2(model, fitness = fitness_func, num_fireflies = 150, alpha=0.4, beta=0.9, gamma=0.001, max_limit=5000)
#history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 100, batch_size=32, verbose = 2)


#Confusion Matrix
typred = history.predict(x_test)
y_pred = np.round(typred)
con_mat = sklearn.metrics.confusion_matrix(y_test, y_pred)
plt.imshow(con_mat, cmap='Pastel1')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['False', 'True'])
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, con_mat[i, j], ha='center', va='center')
plt.show()