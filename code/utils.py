# Investigates the bounds on the rotation unitary entanglement
import torch
import scipy
import itertools 
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from collections import defaultdict

# Vector operations 
vec_overlap = lambda a, b: (a.conj().reshape(1, -1) @ b).abs() ** 2
Adj = lambda x: x.transpose(-1, -2).conj()
outer = lambda vec: kron([vec.reshape(1, -1), Adj(vec.reshape(1, -1))])

# Clip small values to prettify display
def clip_small(tensor, tolerance=1e-7):
    real_part = torch.where(torch.abs(tensor.real) < tolerance,\
                            torch.zeros_like(tensor.real), tensor.real)
    imag_part = torch.where(torch.abs(tensor.imag) < tolerance, \
                            torch.zeros_like(tensor.imag), tensor.imag)
    return torch.complex(real_part, imag_part)

# Given a torch.complex128 matrix, graphically display it 
def analyze(arr, msg=None, real_only=False):
    arr = np.array(arr.detach().cpu().numpy())
    # Ensure the input array is of complex128 data type
    if arr.dtype != np.complex128:
        raise TypeError("Array must be of complex128 data type.")
    
    print(f'{msg} shape {arr.shape}')
    arr = np.matrix(arr)
    # Extract the real and imaginary parts
    real_part = arr.real
    imag_part = arr.imag
    if real_only:
        im1 = plt.imshow(real_part, cmap='viridis')
        # fig.colorbar(im1, ax=axs[0])
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        im1 = axs[0].imshow(real_part, cmap='viridis')
        axs[0].set_title('Real Part')
        fig.colorbar(im1, ax=axs[0])
        im2 = axs[1].imshow(imag_part, cmap='plasma')
        axs[1].set_title('Imaginary Part')
        fig.colorbar(im2, ax=axs[1])
    # Display the plot
    plt.tight_layout()
    plt.show()

# Takes a list of operators and returns their inner product
def kron(X):
    ans = X[0]
    for x in X[1:]: 
        ans = torch.kron(ans.contiguous(), x)
    return ans 


### Qubit operations
def logm(rho):
    if not torch.allclose(rho, Adj(rho)):
        raise RuntimeError('Differentiable logarithm only works with Hermitian input')
    out = torch.linalg.eigh(rho)
    logeigvals = torch.log(out.eigenvalues)
    logeigvals[logeigvals < -10] = 0
    logeigvals = torch.where(torch.isnan(logeigvals), torch.zeros_like(logeigvals), logeigvals)
    return torch.einsum('ij, j, jk->ik', \
            out.eigenvectors, logeigvals, Adj(out.eigenvectors))

def logm_exact(rho):
    return torch.tensor(scipy.linalg.logm(rho.numpy())).type_as(rho)

def entropy(rho):
    return -torch.real(torch.trace(rho @ logm(rho)))


def degree_weight(D):
    ans = defaultdict(lambda: 0)
    for k, v in D.items():
        ans[sum(k)] += v.abs()**2 
    return ans