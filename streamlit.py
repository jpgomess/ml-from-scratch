

import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import numpy as np

def train_and_visualize():
    # Create the neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    # Training data
    X_train = np.asarray([[0.0], [0.2], [0.4], [0.6], [0.8]])
    y_train = np.asarray([[0.0], [0.2], [0.4], [0.6], [0.8]])*2

    fig, axes = plt.subplots(1, 1, figsize=(5,5))
    axes.scatter(X_train, y_train)
    st.pyplot(fig)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training loop
    # epochs = 100
    # for epoch in range(epochs):
    #     model.fit(X_train, y_train, epochs=1, verbose=0)

    #     # Visualize input and output
    #     st.write("Epoch:", epoch + 1)
    #     output = model.predict(X_train[np.random.choice(np.arange(X_train.shape[0]))])
    #     st.write("Output:", output[0][0])

if __name__ == "__main__":
    train_and_visualize()

