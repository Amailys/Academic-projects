import torch
from torch.utils.tensorboard import SummaryWriter
from TP2.tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3,requires_grad=True)
b = torch.randn(3,requires_grad=True)

epsilon = 0.05

writer = SummaryWriter()
u=[]
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)

    y_hat=Linear.apply(x,w,b)
    loss=MSE.apply(y_hat,y)

    writer.add_scalar('Loss/train', loss, n_iter)

    ##  TODO:  Calcul du backward (grad_w, grad_b)

    loss.backward()

    ##  TODO:  Mise à jour des paramètres du modèle
    with torch.no_grad():
        w-=epsilon*w.grad
        b-=epsilon*b.grad

            #remise à zero
    w.grad.zero_()
    b.grad.zero_()
    u.append(loss.item())
writer.close()
