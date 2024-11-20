import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]
    
def embedding(x):
    one_hot=torch.zeros(x.shape[0],x.shape[1],len(id2lettre))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            one_hot[i,j,x[i,j]]=1
    return one_hot     


#  TODO: 

if __name__ == "__main__":
    PATH = "/home/amailys/TP/AMAL/TP3/data/"
    data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)


    from tqdm import tqdm 

    epochs = 5
    epsilon=0.001
    latent=50
    dim_output=96
    dim_input=96
    length=808

    model=RNN(length,dim_input,latent,dim_output)
    cross_entropy=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=epsilon)

    train_loss=[]
    test_loss=[]
    for epoch in tqdm(range(epochs)): 
        
        cumloss=0
        for x,y in data_trump:
            x, y = x.transpose(0, 1), y.transpose(0, 1)
            x=embedding(x)

            h = model(x)
            pred = model.decode(h)
            pred = pred.reshape(-1,  pred.shape[-1])
            y=y.reshape(-1)


            loss = cross_entropy(pred,y)

            cumloss += loss.item() / len(data_trump)

            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss.append(cumloss)

    import matplotlib.pyplot as plt
    plt.plot(range(5),train_loss)
