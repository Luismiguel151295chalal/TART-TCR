import torch
import torch.nn as nn
import torch.nn.functional as F

class TART_TCR_Block(nn.Module):
    def __init__(self, d_model=1024, num_primes=50, noise_p=0.0464):  # M=50 como en tu paper
        super().__init__()
        self.d_model = d_model
        self.M = num_primes
        self.p = noise_p
        
        self.prime_nodes = nn.Parameter(torch.randn(self.M, d_model))
        
        # 3 tipos de bonds explícitos (como describes en sección 3)
        self.w_cov = nn.Parameter(torch.tensor(1.0))
        self.w_hyd = nn.Parameter(torch.tensor(0.5))
        self.w_vdw = nn.Parameter(torch.tensor(0.1))
        
        # Matrices base (mejor inicialización)
        self.A_cov = nn.Parameter(torch.eye(self.M).roll(1, dims=0))  # cyclic shift para sequential
        self.A_hyd = nn.Parameter(torch.tril(torch.ones(self.M, self.M), diagonal=-1))
        self.A_vdw = nn.Parameter(torch.randn(self.M, self.M) * 0.05)  # sparse por defecto
        
        self.expert = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def get_molecular_routing(self):
        raw = self.w_cov * self.A_cov + self.w_hyd * self.A_hyd + self.w_vdw * self.A_vdw
        return F.softmax(raw, dim=-1)

    def forward(self, x, continuous_mode=False, latent_steps=1):
        B, N, D = x.shape
        # TART Collapse
        C = F.softmax(torch.einsum('bnd,md->bnm', F.normalize(x,dim=-1), 
                                   F.normalize(self.prime_nodes,dim=-1)) / 0.1, dim=-1)
        H_prime = torch.einsum('bnm,bnd->bmd', C, x)
        
        # Molecular Routing
        A = self.get_molecular_routing()
        H = torch.einsum('mn,bmd->bmd', A, H_prime)
        
        # TCR + Latent Loop (vectorizado, sin for b)
        for _ in range(latent_steps):
            # SVD orthogonal projection (batch)
            U, _, Vt = torch.linalg.svd(H, full_matrices=False)
            H_logic = torch.bmm(U, Vt)
            
            # ISP 4.64%
            mask = (torch.rand_like(H_logic) < self.p).float()
            noise = torch.randn_like(H_logic) * 0.1
            H = torch.tanh(self.expert(H_logic + mask * noise))
        
        if continuous_mode:
            return H  # solo primes para infinite reasoning
        return x + torch.einsum('bnm,bmd->bnd', C, H)  # residual back to N
