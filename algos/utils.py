import torch
import numpy as np


def generate_fake_keys(key, device, N: int = 32):
    """generates N fake keys that are != to self.key
    and N keys that are close to self.key (one bit away)

    Args:
        N (int, optional): number of keys to generate. Defaults to 32.

    Returns:
        fake_keys, close_keys
    """
    close_keys = key.repeat((N, 1))
    keysize = key.shape[1]
    for i in range(N):
        idx = np.random.randint(0, keysize)
        close_keys[i][idx] = 1 - close_keys[i][idx]
    done = True
    fake_keys = torch.randint(
        0, 2, (N, key.shape[1]), device=device, dtype=torch.float32
    )
    for i in range(N):
        done = done and torch.all(fake_keys[i] != key)
    while not done:
        done = True
        fake_keys = torch.randint(
            0, 2, (N, key.shape[1]), device=device, dtype=torch.float32
        )
        for i in range(N):
            done = done and torch.any(fake_keys[i] != key)
    return fake_keys, close_keys
