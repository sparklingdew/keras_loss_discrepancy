import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

def noise(x_dim, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(x_dim) * noise_level 

def data_generation(x_dim,samples,noise_level):
    X_data=[]
    y_data=[]
    
    for i in range(samples):
        X=np.random.RandomState(i+2).rand(x_dim)
        y=sum([(1/x)**0.5 for x in X])+20*sum([x for x in X])
        X_noise=X+noise(x_dim, noise_level, seed=i+2)
        X_data.append(X_noise.reshape(-1,1))
        y_data.append(y)
    y_data=minmax_scale(y_data)
    X_data=np.array(X_data)
    return X_data,y_data

class WeightsSaver(Callback):
    def __init__(self, N,weights_dir):
        self.N = N
        self.batch = 0
        self.weights_dir=weights_dir

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = self.weights_dir+'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
def set_index_intervals(df,batches,epochs):
    total_epochs=batches*epochs    
    df_index=[*range(batches-1,total_epochs,batches)]
    df.set_index([df_index],inplace=True)
    return df

def evaluation(X_train,y_train,batch_size,batches,epochs,model,weights_dir): 
        #Evaluation from saved weights saved weights after every batch on whole train set
    eval_PerEpoch_OnWholeSet =[]
    for i in range(batches-1,epochs*batches,batches):
        # Loads weights
        model.load_weights(weights_dir+'weights%08d.h5' % i)  
        # Evaluate model
        eval_PerEpoch_OnWholeSet.append(model.evaluate(X_train,y_train, verbose=0)[0])
    eval_PerEpoch_OnWholeSet=set_index_intervals(pd.DataFrame(eval_PerEpoch_OnWholeSet),batches,epochs)
    
    #Evaluation from saved weights saved weights after every batch on whole train set
    eval_PerBatch_OnWholeSet =[]
    for i in range(batches*epochs):
        # Loads weights
        model.load_weights(weights_dir+'weights%08d.h5' % i)  
        # Evaluate model
        eval_PerBatch_OnWholeSet.append(model.evaluate(X_train,y_train, verbose=0)[0])
    eval_PerBatch_OnWholeSet=pd.DataFrame(eval_PerBatch_OnWholeSet)
    
    # Evaluation from saved weights after every batch on each batch
    eval_PerBatch_OnEachBatch =[]
    for i in range(batches*epochs):
        # Loads weights
        model.load_weights(weights_dir+'weights%08d.h5' % i)    
        # Evaluate model
        if i%batches==0:
            b=0
        else:
            b+=batch_size
        eval_PerBatch_OnEachBatch.append(model.evaluate(X_train[b:b+batch_size],y_train[b:b+batch_size], verbose=0)[0])
    eval_PerBatch_OnEachBatch=pd.DataFrame(eval_PerBatch_OnEachBatch)
    
    return eval_PerEpoch_OnWholeSet,eval_PerBatch_OnWholeSet,eval_PerBatch_OnEachBatch

def mean_loss(df,batches,epochs):
    total_epochs=batches*epochs
    mean_df=[df.iloc[b:b+batches].mean() for b in range(0,total_epochs,batches)]
    mean_df=pd.DataFrame(mean_df,index=range(batches-1,total_epochs,batches))
    return mean_df

def plot_loss(batches,epochs,column_names,loss_all,model_dir,model_name):    
    
    #plot train and validation log losses + evaluation loss on whole set
    plt.figure(figsize=(12,12))
    plt.subplot(211)    
    loss_all_dropna=loss_all.dropna()
    plt.scatter(loss_all_dropna.index,loss_all_dropna[column_names[0]], label=column_names[0],color='red')
    plt.scatter(loss_all_dropna.index,loss_all_dropna[column_names[1]], label=column_names[1],color='blue')
    plt.plot(loss_all_dropna[column_names[4]], label=column_names[4],color='grey',linestyle='dashed')
    plt.title('Log and evaluated loss', fontsize=14)
    plt.ylabel('loss', fontsize=12)
    plt.xlabel('epoch x batches', fontsize=12)
    plt.legend(loc='upper right')
      
    #plot train log loss + train log loss in each batch + evaluation loss on each batch
    plt.subplot(212)    
    plt.scatter(loss_all.index,loss_all[column_names[7]], label=column_names[7],color='grey',
            s=100,marker='^')
    plt.scatter(loss_all.index,loss_all[column_names[0]], label=column_names[0],color='red')
    plt.scatter(loss_all.index,loss_all[column_names[3]], label=column_names[3],
                s=100, facecolors='none', edgecolors='g')
    plt.plot(loss_all[column_names[5]], label=column_names[5],color='red', linestyle='dashed')
    plt.plot(loss_all[column_names[6]], label=column_names[6],color='green', linestyle='dashed')
    plt.title('Loss per batch and epoch', fontsize=14)
    plt.ylabel('loss', fontsize=12)
    plt.xlabel('epoch x batches', fontsize=12)
    plt.legend(loc='upper right')
   
    plt.suptitle(model_name+'\n'+str(batches)+' batches - '+str(epochs)+' epochs',fontsize=16,weight='bold')
    
    plt.savefig(model_dir+'/plot_losses', dpi=150)

def plot_prediction(y_pred,y_train,model_dir,model_name,batches):
    
    y_train=y_train.flatten()
    plt.figure()
    plt.plot(y_pred,y_train,'.')
    plt.title('prediction vs training data', fontsize=14)
    plt.ylabel('y_train', fontsize=12)
    plt.xlabel('y_pred', fontsize=12)
    plt.savefig(model_dir+'/plot_prediction', dpi=150)
