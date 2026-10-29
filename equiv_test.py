# Numpy check of the rotation-equivariance argument (Exp. 4), forward pass only.
import numpy as np
rng = np.random.default_rng(0)
C, H = 16, 48
sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)/8
sy = sx.T
ident = np.zeros((3,3)); ident[1,1] = 1

def conv(x, k):            # circular cross-correlation, x:(C,H,W)
    out = np.zeros_like(x)
    for i in range(3):
        for j in range(3):
            out += k[i,j]*np.roll(x, (1-i, 1-j), axis=(1,2))
    return out

def alive(x):
    a = x[3]; m = a.copy()
    for di in (-1,0,1):
        for dj in (-1,0,1):
            m = np.maximum(m, np.roll(a, (di,dj), axis=(0,1)))
    return (m > 0.1).astype(float)

W1 = rng.normal(0, .3, (128, 48)); b1 = rng.normal(0, .1, 128)
W2 = rng.normal(0, .05, (16, 128))
def step(x, theta=0.0):
    c, s = np.cos(theta), np.sin(theta)
    kx, ky = c*sx - s*sy, s*sx + c*sy
    p = np.concatenate([x, conv(x, kx), conv(x, ky)], 0)           # (48,H,W)
    h = np.maximum(np.einsum('oi,ihw->ohw', W1, p) + b1[:,None,None], 0)
    ds = np.einsum('oi,ihw->ohw', W2, h)
    x = x + ds                                                      # fire_rate = 1
    return x * alive(x)

x0 = np.zeros((C,H,H)); x0[3:, H//2-3:H//2+4, H//2-3:H//2+4] = rng.uniform(0.3,1,(13,7,7))
T = 20
def run(x, theta):
    for _ in range(T): x = step(x, theta)
    return x
ref = run(x0, 0.0)
for k in (1, 2, 3):                                  # rotate field by k*90 deg
    xr = np.rot90(x0, k, axes=(1,2))
    for th in (k*np.pi/2, -k*np.pi/2):
        out = run(xr, th)
        err = np.abs(out - np.rot90(ref, k, axes=(1,2))).max()
        print(f"rot90 x{k}, sensor theta={th/np.pi*2:+.0f}*pi/2: max|err| = {err:.2e}")
print("scale of state:", np.abs(ref).max())
