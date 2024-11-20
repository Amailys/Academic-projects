#Do Amailys

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

DATA_PATH = "../tp3-rnn/data/trump_full_speech.txt"

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.

    output=output.view(-1, output.size(-1))
    target=target.view(-1)


    loss=torch.nn.CrossEntropyLoss(reduce="none")

    mask=(target!=padcar)

    loss=torch.nn.CrossEntropyLoss(reduce="none")
    loss(output,target)
    masked_loss=loss*mask
    return masked_loss



class RNN(nn.Module):

    def __init__(self,length,dim,latent,output_size):
        super(RNN, self).__init__()
        self.length = length
        self.dim=dim
        self.latent=latent
        self.output_size = output_size

        self.lx = nn.Linear(self.dim,self.latent)
        self.lh = nn.Linear(self.latent,self.latent)
        self.lh2 = nn.Linear(self.latent,self.output_size)

    def one_step(self,x,h):
        return torch.tanh(self.lh(h)+self.lx(x))

    def forward(self,x,h0=None):
        length,batch=x.shape[0],x.shape[1]
        if h0 == None:
            h0=torch.zeros(batch,self.latent)
        outputs=[]
        h = h0
        
        for i in range(length):
            h = self.one_step(x[i],h)
            outputs.append(h)
        
        outputs=torch.stack(outputs)
        return outputs

    def decode(self,h):
        return torch.sigmoid(self.lh2(h))


class LSTM(RNN):
    def __init__(self,input_size, hidden_size,dictionnaire_size,dim_output):
        super(GRU, self).__init__()
        
        self.hidden_size  = hidden_size
        self.input_size = input_size
        self.latent = hidden_size
        self.embedding=nn.Embeddings(dictionnaire_size,input_size)
        self.dim_output=dim_output

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.lf = nn.Linear(self.input_size+self.latent,self.latent)
        self.li = nn.Linear(self.input_size+self.latent,self.latent)
        self.lc = nn.Linear(self.input_size+self.latent,self.latent)
        self.lo = nn.Linear(self.input_size+self.latent,self.latent)
        self.linear = nn.Linear(self.latent+self.latent,self.latent)
        self.linear_decode=nn.Linear(self.latent,self.dim_output)

    def one_step(self,x,h,Ct):
        ft = self.sigmoid(self.lf(torch.cat((x,h),dim=-1)))
        it = self.sigmoid(self.li(torch.cat((x,h),dim=-1)))
        Ct=ft*Ct+it*self.tanh(self.lc(torch.cat((x,h),dim=-1)))
        ot=self.sigmoid(self.lo(torch.cat((x,h),dim=-1)))
        ht = ot*self.tanh(Ct)
        return ht,Ct
    
    def forward(self,x,h0=None,C0=None):
        x=self.embedding(x)
        length,batch=x.shape[0],x.shape[1]
        if h0 == None:
            h0=torch.zeros(batch,self.latent,device=x.device)
        if C0 == None:
            C0=torch.zeros(batch,self.latent,device=x.device)
        outputs=[]
        h = h0
        Ct=C0
        for i in range(length):
            h,Ct = self.one_step(x[i],h,Ct)
            outputs.append(h)

        outputs=torch.stack(outputs)
        return outputs, Ct,h

    def decode(self,h):
        return torch.sigmoid(self.linear_decode(h))


class GRU(nn.Module):
    def __init__(self,input_size, hidden_size,dictionnaire_size,dim_output):
        super(GRU, self).__init__()
        
        self.input_size = input_size
        self.latent = hidden_size

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dim_output = dim_output
        self.embedding = nn.Embedding(dictionnaire_size,64) # 64 = hyper param


        self.lz = nn.Linear(self.input_size+self.latent,self.latent,bias=False)
        self.lr = nn.Linear(self.input_size+self.latent,self.latent,bias=False)
        self.linear = nn.Linear(self.latent+self.latent,self.latent,bias=False)
        self.linear_decode=nn.Linear(self.latent,self.dim_output)

    def one_step(self,x,h):
        zt = self.sigmoid(self.lz(torch.cat((x,h),dim=-1)))
        rt = self.sigmoid(self.lr(torch.cat((x,h),dim=-1)))
        ht = (1 - zt) * h + zt * self.tanh(self.linear(torch.cat((rt*h,x),dim=-1)))
        return ht
    
    def forward(self,x,h0=None):
        x=self.embedding(x)
        length,batch=x.shape[0],x.shape[1]
        if h0 == None:
            h0=torch.zeros(batch,self.latent,device=x.device)
        outputs=[]
        h = h0
        
        for i in range(length):
            h = self.one_step(x[i],h)
            outputs.append(h)
        
        outputs=torch.stack(outputs)
        return outputs,h
    
    def decode(self,h):
        return torch.sigmoid(self.linear_decode(h))
    

#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import os
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(os.path.expanduser("~"), "runs", f"test_{current_time}")
    writer = SummaryWriter(log_dir=log_dir)

    PATH = "/home/amailys/TP/AMAL/TP3/data/"
    batch_size=32
    raw_data=open(PATH+"trump_full_speech.txt","rb").read().decode()[:5000]
    ds=TextDataset(raw_data,maxlen=1000)
    data_trump = DataLoader(ds, batch_size= batch_size, shuffle=True,collate_fn=pad_collate_fn)

    from tqdm import tqdm 

    epochs = 5
    epsilon=0.001
    latent=50
    dim_output=96
    dim_input=96
    length=808
    cross_entropy=torch.nn.CrossEntropyLoss()

    def training(classe,epochs,epsilon,input_size,dim_output,
                 dictionnaire_size,hidden_size):

        model=classe(input_size, hidden_size,dictionnaire_size,dim_output)
        optimizer=torch.optim.Adam(model.parameters(),lr=epsilon)

        for epoch in tqdm(range(epochs)): 
            
            cumloss=0
            for batch in data_trump:
                batch = batch.transpose(0, 1)
                x=batch[:,:-1]
                y=batch[:,1:]
                h = model(x)
                pred = model.decode(h)
                pred = pred.reshape(-1,  pred.shape[-1])
                y=y.reshape(-1)

                loss = maskedCrossEntropy(pred, y, padcar= 0)

                cumloss += loss.item() / len(data_trump)
                loss.backward() 

                optimizer.step()
                optimizer.zero_grad()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'gradients/{name}', param.grad, epoch)




            
        writer.close()


    training(GRU,5,epsilon,hidden_size=64,input_size=50,dim_output=len(id2lettre),
                 dictionnaire_size=len(id2lettre))
    
    training(LSTM,5,epsilon,hidden_size=64,input_size=50,dim_output=len(id2lettre),
                 dictionnaire_size=len(id2lettre))
