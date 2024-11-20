from utils import RNN, device, ForecastMetroDataset

from torch.utils.data import DataLoader
import torch

# Nombre de stations utilisé
CLASSES = 10
# Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 32

PATH = "/users/nfs/Etu6/21304376/TP/AMAL/TP3/data/"

matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"),weights_only=True)
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

def training(data,model, optimizer,loss_fn,epochs):

    train_loss = []

    for epoch in tqdm(range(epochs)): 
        
        cumloss = 0

        for x, y in data:
            batch, length, nb_stations = x.shape[0], x.shape[1], x.shape[2]
            x, y = x.transpose(0, 1).to(device), y.transpose(0, 1).to(device)
            h_0 = torch.zeros(batch, nb_stations, latent).to(device)
            y_pred = model(x, h_0)
            y_pred = model.decode(y_pred)
            loss = mse(y_pred, y)
            cumloss += loss.item() / len(data)

            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

        train_loss.append(cumloss)

    return train_loss, model

def testing(data,model,loss_fn,epochs):
    test_loss = []

    for epoch in tqdm(range(epochs)): 
        
        cumloss = 0
        with torch.no_grad():
            for x, y in data:
                batch, length, nb_stations = x.shape[0], x.shape[1], x.shape[2]
                x, y = x.transpose(0, 1).to(device), y.transpose(0, 1).to(device) 
                h_0 = torch.zeros(batch, nb_stations, latent).to(device) 
                y_pred = model(x, h_0)
                y_pred = model.decode(y_pred)
                loss = mse(y_pred, y)
                cumloss += loss.item() / len(data)
        
        test_loss.append(cumloss)

    
    return test_loss
    
def prediction(initial_data,model,horizon_prediction,latent):
    model.eval()
    predictions=[]
    x=initial_data.transpose(0,1).to(device)
    for _ in range(horizon_prediction):
        with torch.no_grad():
            length,batch, nb_stations = x.shape[0], x.shape[1], x.shape[2]
            h_0 = torch.zeros(batch, nb_stations, latent).to(device)
            y_pred = model(x, h_0)
            y_pred = model.decode(y_pred)
            f=y_pred[0].unsqueeze(0)
        predictions.append(f)
        x=torch.cat((x[1:,:,:,:],f),dim=0).to(device)
    
    return torch.stack(predictions) 
    

#  TODO:  Question 3 : Prédiction de séries temporelles
if __name__ == "__main__":
    from tqdm import tqdm 
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    epochs = 1
    epsilon = 0.001
    latent = 20
    model = RNN(LENGTH - 1, DIM_INPUT, latent, DIM_INPUT).to(device)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=epsilon)

    training=training(data_train,model, optimizer,mse,epochs)

    for x,y in data_test:
        init_data=x[0]
        init_data=init_data.unsqueeze(0)
        break
    
    pred=prediction(init_data,training[1],23,latent)
    print(pred)
    """
    test=testing(data_test,model,mse,epochs)
    sns.set(style="darkgrid")
    sns.lineplot(x=range(epochs), y=training[0], label='Training Loss')
    sns.lineplot(x=range(epochs), y=test, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.savefig("/users/nfs/Etu6/21304376/TP/AMAL/TP3/loss_plot.png", bbox_inches='tight')

    """

