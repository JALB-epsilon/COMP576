import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################
torch.manual_seed(0)
np.random.seed(0)

################################################################
# Spectral Convolution Layers 
# 1D and 2D
################################################################

class fourier_conv_1d(nn.Module):
  def __init__(self, in_, out_, wavenumber1):
    super(fourier_conv_1d, self).__init__()
    self.in_ = in_
    self.out_ = out_
    self.wavenumber1 = wavenumber1
    self.scale = (1 / (in_ * out_))
    self.weights1 = nn.Parameter(self.scale * torch.rand(in_, out_, self.wavenumber1, dtype=torch.cfloat))
   
    # Complex multiplication
  def compl_mul1d(self, input, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", input, weights)

  def forward(self, x):
    batchsize = x.shape[0]
    #Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = torch.fft.rfft(x)

    # Multiply relevant Fourier modes
    out_ft = torch.zeros(batchsize, self.out_, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
    out_ft[:, :, :self.wavenumber1] = self.compl_mul1d(x_ft[:, :, :self.wavenumber1], self.weights1)

    #Return to physical space
    x = torch.fft.irfft(out_ft, n=x.size(-1))
    return x

  def print(self):
    return f'FourierConv1d({self.in_}, {self.out_}, wavenumber={self.wavenumber1})'

class fourier_conv_2d(nn.Module):
  def __init__(self, in_, out_, wavenumber1, wavenumber2):
    super(fourier_conv_2d, self).__init__()
    self.in_ = in_
    self.out_ = out_
    self.wavenumber1 = wavenumber1
    self.wavenumber2 = wavenumber2
    self.scale = (1 / (in_ * out_))
    self.weights1 = nn.Parameter(self.scale * torch.rand(in_, out_, self.wavenumber1, self.wavenumber2, dtype=torch.cfloat))
    self.weights2 = nn.Parameter(self.scale * torch.rand(in_, out_, self.wavenumber1, self.wavenumber2, dtype=torch.cfloat))

    # Complex multiplication
  def compl_mul2d(self, input, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", input, weights)

  def forward(self, x):
    batchsize = x.shape[0]
    #Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = torch.fft.rfft2(x)
    # Multiply relevant Fourier modes
    out_ft = torch.zeros(batchsize, self.in_,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
    out_ft[:, :, :self.wavenumber1, :self.wavenumber2] = \
        self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2], self.weights1)
    out_ft[:, :, -self.wavenumber1:, :self.wavenumber2] = \
        self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2], self.weights2)
    #Return to physical space
    x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
    return x

  def print(self):
    return f'FourierConv2d({self.in_}, {self.out_}, wavenumber={self.wavenumber1, self.wavenumber2})'

################################################################
# Fourier Layer 
################################################################
class Fourier_layer(nn.Module):
  def __init__(self,  features_, wavenumber, activation = 'relu', is_last = False):
    super(Fourier_layer, self).__init__()
    self.is_last = is_last
    self.activation = activation.lower()
    self.features_ = features_
    self.wavenumber = wavenumber
    self.dim = len(wavenumber)
    self.W =  nn.Conv1d(features_, features_, 1) if self.dim==1 else nn.Conv2d(features_, features_, 1)
    self.fourier_conv = self.set_conv_dim()
    self.nonlinear = set_activ(activation)

  def set_conv_dim(self):
    if self.dim== 1:
      return  fourier_conv_1d(self.features_, self.features_, *self.wavenumber)
    elif self.dim== 2:
      return  fourier_conv_2d(self.features_, self.features_, *self.wavenumber)
   
  def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        x = x1 + x2
        if self.is_last == True:
          return x
        else:
          x = self.nonlinear(x)
          return x
            
  def __repr__(self):
    with torch.no_grad():
      return self.activation+'('+self.fourier_conv.print() +' + '+ self.W.__repr__()+')'

################################################################
# Lifting map
################################################################

class Lifting(nn.Module):
  def __init__(self, input, width, activation = 'relu'):
    super().__init__()
    self.fc1 = nn.Linear(input, width//2)
    self.nonlinear =set_activ(activation)
    self.fc2 = nn.Linear(width//2, width)
  def forward(self,x):
    x = self.fc1(x)
    x = self.nonlinear(x)
    x = self.fc2(x)
    return x

################################################################
# Projection map
################################################################

class Proj(nn.Module):
  def __init__(self,width1, width2=1, activation = 'relu'):
    super().__init__()
    self.fc1 = nn.Linear(width1, 128)
    self.fc2 = nn.Linear(128, width2)
    self.nonlinear =set_activ(activation)
  def forward(self,x):
    x = self.fc1(x)
    x = self.nonlinear(x)
    x = self.fc2(x)
    return x
################################################################
# FNO map 1D and 2D
################################################################

class FNO(nn.Module):
  def __init__(self, wavenumber, features_, layers, 
                    padding = 9, 
                    activation= 'relu',
                    lifting = Lifting, 
                    proj = Proj):
    super(FNO, self).__init__()
    self.wavenumber = wavenumber
    self.dim = len(wavenumber)
    self.activation = activation.lower() 
    self.padding = padding   
    self.features_ =features_
    self.layers = layers
    self.lifting = lifting(self.dim+1, self.features_)
    self.fno = []
    self.proj = proj(self.features_, 1)

    for l in range(layers-1):
      self.fno.append(Fourier_layer(features_ = self.features_, 
                                    wavenumber=self.wavenumber, 
                                    activation = self.activation))
      
    self.fno.append(Fourier_layer(features_=self.features_, 
                                    wavenumber=self.wavenumber, 
                                    activation = self.activation,
                                    is_last= True))
    self.fno =nn.Sequential(*self.fno)

  def forward(self, x):
    grid = self.get_grid2D(x.shape, x.device) if self.dim == 2 else self.get_grid1D(x.shape, x.device)
    x = torch.cat((x, grid), dim=-1)
    ####Lifting Map 
    x = self.lifting(x)
    ###Actual Neural Operator
    x = x.permute(0, 3, 1, 2) if self.dim == 2 else x.permute(0, 2, 1)
    x = F.pad(x, [0,self.padding, 0,self.padding]) if self.dim == 2 else F.pad(x, [0,self.padding])
    x = self.fno(x)
    x = x[..., :-self.padding, :-self.padding] if self.dim == 2 else x[..., :-self.padding]
    x = x.permute(0, 2, 3, 1) if self.dim == 2 else x.permute(0, 2, 1)
    ####Projection Map
    x =self.proj(x)
    return x
    
  def get_grid2D(self, shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)

  def get_grid1D(self, shape, device):
    batchsize, size_x = shape[0], shape[1]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
    return gridx.to(device)

################################################################
# Miscellaneous
################################################################

def set_activ(activation):
  with torch.no_grad():
    if activation == 'relu':
      nonlinear = F.relu
    elif activation == 'tanh':
      nonlinear = nn.Tanh()
    elif activation == 'sine':
      nonlinear= torch.sin
    elif activation == 'gelu':
      nonlinear= F.gelu
    elif activation == None:
      nonlinear = lambda a : a
    else:
      raise Exception('The activation is not recognized from the list')
    return nonlinear

##########################################################