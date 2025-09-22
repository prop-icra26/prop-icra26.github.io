from argparse import ArgumentParser
from algos.networks import MLPK, MLP, CVAE
from algos.utils import generate_fake_keys
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
import tqdm
import os
import pickle
import sys


def generate_data(N, dim: int = 2):
    action_space = np.random.uniform(-1.0, 1.0, (1000, dim))
    action_space[0] *= 0.0
    states = np.random.uniform(-10.0, 10.0, (N, 2 * dim))
    actions = np.empty((N, dim))
    actions_aug = np.empty((N, dim))
    for i in range(N):
        best_cost = np.inf
        best_action = action_space[0]
        for action in action_space:
            cost = np.linalg.norm(states[i][0:dim] + action - states[i][dim : 2 * dim])
            if cost < best_cost:
                best_cost = cost
                best_action = action
        actions[i] = best_action
        actions_aug[i] = -best_action
    return states, actions, actions_aug


def main(args):
    device = torch.device("cpu")
    mlp = MLPK(2 * args.dim, args.keysize, args.dim, args.keysize, 2).to(device)
    print(
        sum(p.numel() for p in mlp.parameters() if p.requires_grad_),
        "parameters in our mlp",
    )
    mlp.train()
    states, actions, actions_key = generate_data(args.N_datapoints, dim=args.dim)
    states_t = torch.tensor(states, device=device, dtype=torch.float)
    actions_t = torch.tensor(actions, device=device, dtype=torch.float)
    actions_key_t = torch.tensor(actions_key, device=device, dtype=torch.float)
    cutoff = int(0.8 * args.N_datapoints)
    train_input = states_t[0:cutoff]
    train_output = actions_t[0:cutoff]
    train_output_k = actions_key_t[0:cutoff]
    valid_input = states_t[cutoff:]
    valid_output = actions_t[cutoff:]
    valid_output_k = actions_key_t[cutoff:]

    # begin training (ours)
    key = torch.randint(0, 2, (1, args.keysize), device=device, dtype=torch.float)
    train_key = key.repeat((len(train_input), 1))
    valid_key = key.repeat((len(valid_input), 1))
    optim = AdamW(mlp.parameters(), lr=args.lr)
    tbar = tqdm.trange(args.epochs, file=sys.stdout)
    for e in tbar:
        fake_keys, close_keys = generate_fake_keys(key, device, N=cutoff)
        output_kstar = mlp.forward(train_input, train_key)
        output_fake = mlp.forward(train_input, fake_keys)
        output_close = mlp.forward(train_input, close_keys)
        loss1 = F.mse_loss(output_kstar, train_output_k)
        loss2 = F.mse_loss(output_fake, train_output)
        loss3 = F.mse_loss(output_close, train_output)
        loss = loss1 + loss2 + loss3
        optim.zero_grad()
        loss.backward()
        optim.step()
        tbar.set_description(f"OURS: Losses: {loss1.item():3.3f} {loss2.item():3.3f} {loss3.item():3.3f}")
        if e % (args.epochs // 10) == 0 and e != 0:
            # print validation loss
            mlp.eval()
            with torch.no_grad():
                fake_keys, close_keys = generate_fake_keys(key, device, N=len(valid_input))
                output_kstar = mlp.forward(valid_input, valid_key)
                output_fake = mlp.forward(valid_input, fake_keys)
                output_close = mlp.forward(valid_input, close_keys)
                loss1 = F.mse_loss(output_kstar, valid_output_k)
                loss2 = F.mse_loss(output_fake, valid_output)
                loss3 = F.mse_loss(output_close, valid_output)
                print(f"OURS: Validation Losses: {loss1.item():3.3f} {loss2.item():3.3f} {loss3.item():3.3f}")
            mlp.train()
    tbar.close()
    fake_keys, close_keys = generate_fake_keys(key, device, N=len(valid_input))
    output_kstar = mlp.forward(valid_input, valid_key)
    output_fake = mlp.forward(valid_input, fake_keys)
    output_close = mlp.forward(valid_input, close_keys)
    ours_loss1 = F.mse_loss(output_kstar, valid_output_k)
    ours_loss2 = F.mse_loss(output_fake, valid_output)
    ours_loss3 = F.mse_loss(output_close, valid_output)
    print(f"OURS: Final Validation Losses: {ours_loss1.item():3.3f} {ours_loss2.item():3.3f} {ours_loss3.item():3.3f}")
    # baseline:
    baseline_mlp = MLP(2 * args.dim + args.keysize, args.keysize, args.dim, 4).to(device)
    print(
        sum(p.numel() for p in baseline_mlp.parameters() if p.requires_grad_),
        "parameters in baseline_mlp",
    )
    optim = AdamW(baseline_mlp.parameters(), lr=args.lr)
    # begin training (baseline)
    tbar = tqdm.trange(args.epochs, file=sys.stdout)
    for e in tbar:
        fake_keys, close_keys = generate_fake_keys(key, device, N=cutoff)
        output_kstar = baseline_mlp.forward(torch.concat((train_input, train_key), dim=1))
        output_fake = baseline_mlp.forward(torch.concat((train_input, fake_keys), dim=1))
        output_close = baseline_mlp.forward(torch.concat((train_input, close_keys), dim=1))
        loss1 = F.mse_loss(output_kstar, train_output_k)
        loss2 = F.mse_loss(output_fake, train_output)
        loss3 = F.mse_loss(output_close, train_output)
        loss = loss1 + loss2 + loss3
        optim.zero_grad()
        loss.backward()
        optim.step()
        tbar.set_description(f"BASE: Losses: {loss1.item():3.3f} {loss2.item():3.3f} {loss3.item():3.3f}")
        if e % (args.epochs // 10) == 0 and e != 0:
            # print validation loss
            baseline_mlp.eval()
            with torch.no_grad():
                fake_keys, close_keys = generate_fake_keys(key, device, N=len(valid_input))
                output_kstar = baseline_mlp.forward(torch.concat((valid_input, valid_key), dim=1))
                output_fake = baseline_mlp.forward(torch.concat((valid_input, fake_keys), dim=1))
                output_close = baseline_mlp.forward(torch.concat((valid_input, close_keys), dim=1))
                loss1 = F.mse_loss(output_kstar, valid_output_k)
                loss2 = F.mse_loss(output_fake, valid_output)
                loss3 = F.mse_loss(output_close, valid_output)
                print(f"BASE: Validation Losses: {loss1.item():3.3f} {loss2.item():3.3f} {loss3.item():3.3f}")
            baseline_mlp.train()
    tbar.close()
    fake_keys, close_keys = generate_fake_keys(key, device, N=len(valid_input))
    output_kstar = baseline_mlp.forward(torch.concat((valid_input, valid_key), dim=1))
    output_fake = baseline_mlp.forward(torch.concat((valid_input, fake_keys), dim=1))
    output_close = baseline_mlp.forward(torch.concat((valid_input, close_keys), dim=1))
    base_loss1 = F.mse_loss(output_kstar, valid_output_k)
    base_loss2 = F.mse_loss(output_fake, valid_output)
    base_loss3 = F.mse_loss(output_close, valid_output)
    print(f"BASE: Final Losses: {base_loss1.item():3.3f} {base_loss2.item():3.3f} {base_loss3.item():3.3f}")

    """vae section"""
    vae = CVAE(args.keysize, 16, 2 * args.dim, args.dim).to(device)
    print(
        sum(p.numel() for p in vae.parameters() if p.requires_grad_),
        "parameters in cvae",
    )
    optim = AdamW(vae.parameters(), lr=args.lr)
    # begin training (cvae)
    
    """ sagar please save me from this pain """
    tbar = tqdm.trange(args.epochs, file=sys.stdout)
    for e in tbar:
        fake_keys, close_keys = generate_fake_keys(key, device, N=cutoff)
        output_kstar = vae.forward(train_key, train_input)
        output_fake = vae.forward(fake_keys, train_input)
        output_close = vae.forward(close_keys, train_input)
        """i tried both order of inputs (key as input vs. key as latent condition) and nothing works well"""
        loss1 = F.mse_loss(output_kstar, train_output_k) 
        loss2 = F.mse_loss(output_fake, train_output)
        loss3 = F.mse_loss(output_close, train_output)
        # loss = loss1 + loss2 + loss3 + (vae.kld(train_key) + vae.kld(fake_keys) + vae.kld(close_keys)) / 3.0
        loss = loss1 + loss2 + loss3
        optim.zero_grad()
        loss.backward()
        optim.step()
        tbar.set_description(f"cVAE: Losses: {loss1.item():3.3f} {loss2.item():3.3f} {loss3.item():3.3f}")
        if e % (args.epochs // 10) == 0 and e != 0:
            # print validation loss
            vae.eval()
            with torch.no_grad():
                fake_keys, close_keys = generate_fake_keys(key, device, N=len(valid_input))
                output_kstar = vae.forward(valid_key, valid_input)
                output_fake = vae.forward(fake_keys, valid_input)
                output_close = vae.forward(close_keys, valid_input)
                loss1 = F.mse_loss(output_kstar, valid_output_k)
                loss2 = F.mse_loss(output_fake, valid_output)
                loss3 = F.mse_loss(output_close, valid_output)
                print(f"cVAE: Validation Losses: {loss1.item():3.3f} {loss2.item():3.3f} {loss3.item():3.3f}")
            vae.train()
    tbar.close()
    fake_keys, close_keys = generate_fake_keys(key, device, N=len(valid_input))
    output_kstar = vae.forward(valid_key, valid_input)
    output_fake = vae.forward(fake_keys, valid_input)
    output_close = vae.forward(close_keys, valid_input)
    vae_loss1 = F.mse_loss(output_kstar, valid_output_k)
    vae_loss2 = F.mse_loss(output_fake, valid_output)
    vae_loss3 = F.mse_loss(output_close, valid_output)
    print(f"cVAE: Final Losses: {vae_loss1.item():3.3f} {vae_loss2.item():3.3f} {vae_loss3.item():3.3f}")
    if args.save_to_file:
        os.makedirs(args.dirname, exist_ok=True)
        torch.save(mlp.state_dict(), args.dirname + "/ours_weights.pt")
        torch.save(baseline_mlp.state_dict(), args.dirname + "/baseline_mlp_weights.pt")
        torch.save(vae.state_dict(), args.dirname + "/cvae_weights.pt")
        with open(args.dirname + "/dataset.pkl", "wb") as fh:
            pickle.dump(
                {
                    "states": states,
                    "actions": actions,
                    "actions_key": actions_key,
                    "losses": [
                        [ours_loss1.item(), ours_loss2.item(), ours_loss3.item()],
                        [base_loss1.item(), base_loss2.item(), base_loss3.item()],
                        [vae_loss1.item(), vae_loss2.item(), vae_loss3.item()],
                    ],
                    "key": key.cpu().detach().numpy(),
                },
                fh,
            )

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--no-print", action="store_true", default=False)
    parser.add_argument("--save-to-file", action="store_true", default=False)
    parser.add_argument("--dirname", type=str, default="MOVE_ME_I_WILL_BE_OVERWRITTEN")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--N-datapoints", type=int, default=2048)
    parser.add_argument("--keysize", type=int, default=96)
    args = parser.parse_args()
    if args.no_print:
        f = open("/dev/null", "w")
        sys.stdout = f
        main(args)
        f.close()
    else:
        main(args)
