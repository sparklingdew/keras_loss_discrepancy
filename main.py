import os
import math

from regression_models import model_processing
from utils import data_generation

# In['data generation']:

x_dim=50
samples=50
noise_level=1.0
X_data,y_data=data_generation(x_dim,samples,noise_level)

X_train=X_data
y_train=y_data


# In['model']:

epochs_list=[5,50]
batches_list=[1,10,samples]
batch_size_list=list([math.ceil(samples/b) for b in batches_list])
curr_dir=os.getcwd() + '/'
model_names=['MLP','CNN','CNN_batch_norm','CNN_dropout']

epochs_list=[50]
batches_list=[5]
batch_size_list=list([math.ceil(samples/b) for b in batches_list])
model_names=['CNN_batch_norm','CNN_dropout']
iterations=5

for iter in range(iterations):
    for model_name in model_names:
        for epochs in epochs_list:
            for batch_size,batches in zip(batch_size_list,batches_list):
                model_dir=curr_dir+'Iteration_'+str(iter)+'/'+model_name+'_batches_'+str(batches)+'_epochs_'+str(epochs)
                loss=model_processing(X_train,y_train,x_dim,batch_size,batches,epochs,model_dir,model_name)


