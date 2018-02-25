# LSTM / GRU / RNN for Financial Time-Series Prediction

Tony Tong (taotong@berkeley.edu, ttong@pro-ai.org)

### lstm.py
A high-level multi-layer LSTM recurrent neural network interface tailored for financial time-series prediction built on top of TensorFlow backend. 

<img src="multilayer_lstm.jpg" alt="lstm" align="middle" width=70%>

#### LSTM network

```python
# Build LSTM network
g = LSTM(n_input_features=10, batch_size=None, n_states=50, n_layers=2, n_time_steps=100, 
         l1_reg=0.05, l2_reg=0.01, start_learning_rate=0.001, decay_steps=1, decay_rate=0.3, 
         inner_iteration=10, forward_step=1)

# Using a generator data feeder
results = g.train(data_feeder=generate_training_data, epoch_end=50, 
                  display_step=1, return_weights=True)

# Using direct batch data
results = g.train(batch_X=batch_X, batch_y=batch_y, in_sample_size=1600, epoch_end=50, 
                  display_step=1, return_weights=True)
```

#### GRU network

```python
g = GRU(n_input_features=10, batch_size=None, n_states=50, n_layers=2, n_time_steps=100, 
        l1_reg=0.05, l2_reg=0.01, start_learning_rate=0.001, decay_steps=1, decay_rate=0.3, 
        inner_iteration=10, forward_step=1)
```

#### Basic RNN network

```python
g = RNN(n_input_features=10, batch_size=None, n_states=50, n_layers=2, n_time_steps=100, 
        l1_reg=0.05, l2_reg=0.01, start_learning_rate=0.001, decay_steps=1, decay_rate=0.3, 
        inner_iteration=10, forward_step=1)
```

Show tensorflow graph:

```python
g.show_graph()
```

<img src="tensorflow_schematic.jpg" alt="lstm" class="center" width=95%>
