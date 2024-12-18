
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        return (1/y.shape[0])*torch.norm(yhat-y)**2

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)

        u=(2/y.shape[0])
        return u*grad_output*(yhat-y),-u*grad_output*(yhat-y) 

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE
class Linear(Function):
    @staticmethod
    def forward(ctx, X,W,b ):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W,b)

        #  TODO:  Renvoyer la valeur de la fonction
        return X@W+b

    @staticmethod
    def backward(ctx, grad_output):
        X,W,b = ctx.saved_tensors

        return grad_output@W.T,X.T@grad_output,grad_output.sum(0) 
    
## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

