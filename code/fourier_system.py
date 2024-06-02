from utils import * 

'''
    Super-class for the representation of freely generated algebra. 
    Algebra elements are defined as generators, as well as inner products. 
'''

# Pauli matrices
Y = torch.tensor(np.array([[0, -1j], [1j, 0]]))
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)

class FourierSystem:
    def __init__(self, n, dev, cache):
        self.n = n 
        self.dev = dev 
        self.cache = cache
        self.init_generators()
        self.basis_elms = {}

    def init_generators(self): 
        # Initializes self.generators, self.bindex, and self.d
        # The inheriting class should specify 
        #  - self.generators 
        #  - self.d (used for normalizing inner product)
        #  - self.bindex (index for how generators generate basis elements)
        raise NotImplementedError
    
    def id(self): 
        # Identity element 
        raise NotImplementedError

    def __call__(self, k):
        # Shorthand for accessing a generator
        # Generators are 1-indexed, with 0 denoting identity
        if k == 0:
            return self.id()
        return self.generators[k - 1]
    
    def basis_elm(self, J):
        # J is a binary list of length 
        # Basis elements are lazily initialized upon request and cached
        J = tuple(J)
        if J in self.basis_elms.keys():
            return self.basis_elms[J]
        result = self.id()
        if len(J) != len(self.generators):
            raise RuntimeError(f'Expected {len(self.generators)} entries for J={J}')
        for i, j in enumerate(J):
            if j == 1:
                result = result @ self.generators[i]
        if self.cache:
            self.basis_elms[J] = result 
        return result 

    def overlap(self, a, b):
        # Normalized overlap
        result = torch.trace(Adj(a) @ b)
        return clip_small(result) / self.d
    
    def fourier_coeffs(self, rho):
        # Computes the projection coefficients upon generators
        result = {}
        for index in self.bindex:
            c = self.overlap(self.basis_elm(index), rho)
            if c != 0:
                result[index] = c 
        return result 
    
    def assemble(self, coeffs):
        # Assembling an algebra element based on expansion 
        result = None
        for index, coeff in coeffs.items():
            if result is None: 
                result = self.basis_elm(index) * coeff 
            else:
                result += self.basis_elm(index) * coeff 
        return result
    
    
    
    
binstr = lambda n: list(map(lambda x:list(x)[::-1], itertools.product([0, 1], repeat=n)))

# Representation of the majorana algebra over 2^n dimensional Hilbert space
class CliffordAlgebra(FourierSystem):
    def __init__(self, n, dev, cache=True): # Clifford algebra on 2n generators
        super().__init__(n, dev, cache) 

    def id(self):
        return torch.eye(2**self.n, dtype=torch.complex128).to(self.dev) # kron([I]*self.n)
    
    def init_generators(self):
        self.generators = []
        Y = torch.tensor(np.array([[0, -1j], [1j, 0]])).to(self.dev)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128).to(self.dev)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128).to(self.dev)
        I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128).to(self.dev)
        x = lambda i, n: kron([Z]*(i-1) + [X] + [I]*(n-i))
        p = lambda i, n: kron([Z]*(i-1) + [Y] + [I]*(n-i))
        for i in range(1, self.n+1):
            self.generators.append(x(i, self.n))
            self.generators.append(p(i, self.n))
        self.bindex = list(map(tuple, binstr(len(self.generators))))
        self.d = 2**self.n

    def generate_hamiltonian(self, H_coeffs):
        # Given a dictionary of real coefficients
        # assembles the corresponding Hermitian operator after 
        # suitably multiplying by i (or 1) to maintain Hermiticity. 
        H = None 
        for k, v in H_coeffs.items():
            multiplier = 1 if sum(k) % 4  <= 1 else 1j 
            if H is None: 
                H = multiplier * v * self.basis_elm(k)
            else: 
                H += multiplier * v * self.basis_elm(k)
        return H 
    
    def generate_unitary(self, H_coeffs):
        H = self.generate_hamiltonian(H_coeffs)
        return torch.matrix_exp(1j * H)

    def gibbs_state(self, H_coeffs):
        H = self.generate_hamiltonian(H_coeffs)
        rho = torch.matrix_exp(H)
        return rho / torch.trace(rho)

    # def moments(self, rho):
    #     result = {}
    #     for index in self.bindex:
    #         c = self.overlap(Adj(self.basis_elm(index)), rho) * self.d
    #         if c != 0: 
    #             result[index] = c 
    #     return result


# Representation of the Grassmann algebra over 2n generators on 4**n dimensions
#   Conjugate relations are not faithfully represented
class GrassmanAlgebra(FourierSystem):
    def __init__(self, n, dev, cache=True): # Grassmann algebra on 2n generators
        super().__init__(n, dev, cache) 

    def id(self):
        return torch.eye(4**self.n, dtype=torch.complex128).to(self.dev)
    
    def init_generators(self):
        self.generators = []
        sigma_plus = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128).to(self.dev) * (2**.5)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128).to(self.dev)
        I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128).to(self.dev)
        for k in range(1, 2*self.n+1):
            self.generators.append(kron([Z]*(k-1) + [sigma_plus] + [I]*(2*self.n - k))) 
        self.bindex = list(map(tuple, binstr(len(self.generators))))
        self.d = 4**self.n
    
    # Grassmann logarithms converge after a finite number of terms
    def log(self, X):
        if len(X) != 4**self.n:
            print(f'For G(2*{self.n}), expecting size {4**self.n} not {X.shape}')
        c = self.overlap(self.id(), X)
        if (c.abs() < 1e-5):
            raise RuntimeError('Logarithm will not converge without identity component')
        Q = X / c - self.id() 
        result = torch.zeros_like(X) 
        for j in range(1, 2*self.n+1):
            result += torch.matrix_power(Q, j) / j * (-1)**(j-1)
        return result + torch.log(c) * self.id()

# n-qubit system with 2n Clifford / Grassmann generators 
# Handles the conversion between the two sytems 

class MixedAlgebra:
    def __init__(self, n, dev):
        self.C = CliffordAlgebra(self.n, self.dev)
        self.G = GrassmanAlgebra(self.n, self.dev)

    def fourier_transform(self, clifford_op):
        return self.G.assemble({k:v for k, v in self.C.fourier_coeffs(clifford_op).items()})
    
    def inverse_fourier_transform(self, grassmann_op):
        self.C.assemble({k:v for k, v in self.G.fourier_coeffs(grassmann_op).items()})

    def moments(self, rho):
        assert(len(rho) == 2**self.n)
        return self.C.fourier_coeffs(rho)

    def cumulants(self, rho):
        return self.G.fourier_coeffs(self.G.log(self.fourier_transform(rho)))
    
    def magic_norm(self, rho):
        assert(len(rho) == 2**self.n)
        ans = 0
        F = self.cumulants(rho)
        for k, v in F.items():
            if sum(k) > 2:
                ans += torch.abs(v)**2 
        return ans 
    

def rotation_unitary(n, dev='cpu', theta=np.pi/4):
    C = CliffordAlgebra(2*n, dev)
    H = torch.zeros((4**n, 4**n)).to(torch.complex128).to(dev)
    for i in range(2*n):
        H += theta * .5 * C.generators[i] @ C.generators[2*n + i]
    return torch.matrix_exp(H)

def eveniter(n): # Returns all length-n bitstrings with even weight
    return list(filter(lambda x: sum(x) % 2 == 0, binstr(n)))

class EvenConvolution:
    def __init__(self, n, dev, theta=np.pi/4):
        self.rot = rotation_unitary(n, dev, theta)
        
    def apply(self, rho, sigma):
        out = self.rot @ kron([rho, sigma]) @ Adj(self.rot)
        d = rho.shape[0]
        return torch.einsum('abcb->ac', out.reshape(d, d, d, d))


# Converts between index and mask representations of multi-indices
# Adds support for mask pairs
class IndexMaskCvt:
    def __init__(self, n):
        self.n = n 
        
    # Given a number representation, convert to mask
    def n2m(self, x):
        if len(x) == 0:
            return tuple([0]*self.n)
        assert len(x) <= self.n
        assert max(x) <= self.n
        assert min(x) >= 1
        out = [0]*self.n 
        for i in x:
            out[i - 1] = 1
        return tuple(out)
    
    # Given a pair of number representations, convert to mask 
    def n22m(self, x1, x2):
        return tuple(list(self.n2m(x1)) + list(self.n2m(x2)))
    
    def m2n(self, m):
        assert len(m) == self.n
        result = []
        for i in range(self.n):
            if m[i] == 1:
                result.append(i+1)
        return tuple(result)
    
    def m22n(self, m):
        return self.m2n(m[:self.n]), self.m2n(m[self.n:])
    
# Pretty-prints a coefficient dictionary
def pp(coeffs):
    if len(coeffs) == 0:
        return 
    n = len(list(coeffs.keys())[0])
    I = IndexMaskCvt(n)
    for k, v in coeffs.items():
        v = clip_small(v, tolerance=1e-5)
        if v != 0:
            print(f'{I.m2n(k)}, {v.item():.4f}')


            
# def pp2(coeffs): # Pretty-print
#     n = len(list(coeffs.keys())[0]) // 2
#     I = IndexMaskCvt(n)
#     for k, v in coeffs.items():
#         v = clip_small(v, tolerance=1e-5)
#         if v != 0:
#             print(f'{I.m22n(k)}, {v.item():.4f}')