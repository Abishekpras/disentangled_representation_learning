from torch import nn

class beta_vae(nn.Module):
	def __init__(self, D_in, H, D_out):
		super(beta_vae, self).__init__()
		self.enc = nn.Linear(D_in, H)
		self.dec = nn.Linear(H, D_out)

	def forward(self, x):
		latent = self.enc(x)
		return self.dec(latent)
		