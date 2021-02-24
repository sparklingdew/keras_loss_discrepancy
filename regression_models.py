import os
import shutil
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

from utils import WeightsSaver
from utils import LossHistory
from utils import evaluation
from utils import plot_loss
from utils import plot_prediction
from utils import set_index_intervals
from utils import mean_loss

def MLP (x_dim):
    model = Sequential()
    model.add(keras.Input(shape=(x_dim,1)))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))    
    return model

def CNN(x_dim):
    model = Sequential()
    model.add(keras.Input(shape=(x_dim,1)))
    model.add(Conv1D(filters=4, kernel_size=4,strides=5, input_shape=(x_dim,1), use_bias=False))
    model.add(Activation("relu"))
    model.add(Conv1D(filters=4, kernel_size=2,strides=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def CNN_batch_norm(x_dim):
    model = Sequential()
    model.add(keras.Input(shape=(x_dim,1)))
    model.add(Conv1D(filters=4, kernel_size=4,strides=5, input_shape=(x_dim,1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv1D(filters=4, kernel_size=2,strides=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def CNN_dropout(x_dim):
    model = Sequential()
    model.add(keras.Input(shape=(x_dim,1)))
    model.add(Conv1D(filters=4, kernel_size=4,strides=5, input_shape=(x_dim,1), use_bias=False))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Conv1D(filters=4, kernel_size=2,strides=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def model_processing (X_train,y_train,x_dim,batch_size,batches,epochs,model_dir,model_name):
    
    #build
    model=eval(model_name)
    model=model(x_dim)
    
    # compile
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt, metrics=['mean_squared_error'])
    
    #create folder where to save weights. It is deleted in every re-run.
    weights_dir=model_dir+'/weights/'
    try:
        shutil.rmtree(weights_dir)
        os.makedirs(weights_dir)
    except:
        os.makedirs(weights_dir)
        
    #callbacks
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.001)
    batches_period=1 #save weights every 1 batch
    loss_hist=LossHistory()
    weight_saver=WeightsSaver(batches_period,weights_dir)
    callbacks=[reduce_lr, weight_saver,loss_hist]
    
    # fit
    history_PerEpoch = model.fit(X_train,y_train,
                              validation_data=(X_train,y_train),
                              epochs=epochs,callbacks=callbacks,shuffle=False,
                              batch_size=batch_size)
    
    # log metrics and evaluation
    history_PerEpoch=set_index_intervals(pd.DataFrame(history_PerEpoch.history),batches,epochs)
    history_PerBatch=pd.DataFrame(loss_hist.losses)
    eval_PerEpoch_OnWholeSet,eval_PerBatch_OnWholeSet,eval_PerBatch_OnEachBatch=evaluation(X_train,y_train,batch_size,batches,
                                                                  epochs,model,weights_dir)
    mean_loss_per_epoch=mean_loss(eval_PerBatch_OnEachBatch,batches,epochs)
    
    column_names=['log per epoch (train)','log per epoch (val)','learning rate','log per batch (train)',
                  'eval per epoch','eval per batch on whole training set', 'eval per batch on each corresponding batch',
                  'eval per batch on each corresponding batch (mean per epoch)']
    loss_all = pd.concat([history_PerEpoch[['loss','val_loss','lr']],
                    history_PerBatch,
                    eval_PerEpoch_OnWholeSet,
                    eval_PerBatch_OnWholeSet,
                    eval_PerBatch_OnEachBatch,
                    mean_loss_per_epoch],
                    axis=1, ignore_index=False)
    loss_all.columns=column_names
    loss_all.to_csv(model_dir+'/loss_values.csv')
    
    # prediction
    y_pred=model.predict(X_train)
    
    # plot
    plot_loss(batches,epochs,column_names,loss_all,model_dir,model_name)
    
    plot_prediction(y_pred,y_train,model_dir,model_name,batches)
    
    
    # reset weights
    keras.backend.clear_session()

    return loss_all