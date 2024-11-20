from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "/home/amailys/TP/AMAL/TP3/data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
if __name__ == "__main__":
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score 

    epochs = 50
    epsilon=0.0001
    latent=100
    model=RNN(LENGTH,DIM_INPUT,latent,10)
    mse=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=epsilon)

    train_loss=[]
    test_loss=[]
    train_acc=[]
    test_acc=[]

    for epoch in tqdm(range(epochs)): 
        cumloss=0
        accuracy1=0
        i=0
        for data in data_train: 
            input,label=data
            output=model(input)
            output=model.decode(output)

            loss=mse(output,label)
            cumloss+=loss.item()

            accuracy1+=torch.sum(torch.argmax(output,dim=1)==label)
            i+=len(label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        cumloss2=0
        accuracy2=0
        j=0
        with torch.no_grad():
            for data in data_test: 
                input,label=data
                output=model(input)
                output=model.decode(output)

                loss=mse(output,label)
                cumloss2+=loss.item()

                accuracy2+=torch.sum(torch.argmax(output,dim=1)==label)
                j+=len(label)

        train_loss.append(cumloss/i)
        test_loss.append(cumloss2/j)

        train_acc.append(accuracy1/i)
        test_acc.append(accuracy2/j)

    import matplotlib.pyplot as plt

    fig,(ax1,ax2)=plt.subplots(nrows=2,ncols=1,figsize=(10,5))
    ax1.plot(range(epochs),train_loss,label="train loss")
    ax1.plot(range(epochs),test_loss,label="test loss")
    ax1.legend()

    ax2.plot(range(epochs),train_acc,label="train acc")
    ax2.plot(range(epochs),test_acc,label="test acc")
    ax2.legend();