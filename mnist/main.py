from argparse import ArgumentParser
from algos.networks import MNISTClassifier
from algos.baselines import Baseline_MNISTClassifier
from algos.utils import generate_fake_keys
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
import tqdm
import os
import pickle
from torchvision import datasets, transforms
import sys
from torch.utils.data import DataLoader


def main(args):
    """in this version,
    we just try to match to a random offset label
    (e.g., (label + 4) % 10)
    """
    offset_label = np.random.randint(1, 10)
    personalize = lambda label: (label + offset_label) % 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MNISTClassifier(args.keysize).to(device)
    print(
        sum(p.numel() for p in classifier.parameters() if p.requires_grad_),
        "parameters in our cnn",
    )
    trainset = datasets.MNIST(
        "datasets",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    validset = datasets.MNIST(
        "datasets",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
    trainloader = DataLoader(trainset, shuffle=True, batch_size=args.N_datapoints)
    validloader = DataLoader(validset, shuffle=True, batch_size=args.N_datapoints)
    optim = AdamW(classifier.parameters(), args.lr)
    label_to_one_hot = torch.eye(10).to(device)
    key = torch.randint(0, 2, (1, args.keysize), device=device, dtype=torch.float)
    classifier.train()
    tbar = tqdm.trange(args.epochs, file=sys.stdout)
    for e in tbar:
        l1 = 0.0
        l2 = 0.0
        l3 = 0.0
        for image, label in trainloader:
            k_fake, k_close = generate_fake_keys(key, device, N=len(image))
            image_t = image.to(device)
            label_t = label_to_one_hot[label]
            label_k = label_to_one_hot[personalize(label)]
            pred_k = classifier.forward(image_t, torch.tile(key, (len(image), 1)))
            pred_close = classifier.forward(image_t, k_close)
            pred_fake = classifier.forward(image_t, k_fake)
            loss1 = F.cross_entropy(pred_k, label_k)
            loss2 = F.cross_entropy(pred_fake, label_t)
            loss3 = F.cross_entropy(pred_close, label_t)
            loss = 5 * loss1 + loss2 + loss3
            optim.zero_grad()
            loss.backward()
            optim.step()
            l1 += loss1.item()
            l2 += loss2.item()
            l3 += loss3.item()
        tbar.set_description(f"OURS: {l1:3.3f} {l2:3.3f} {l3:3.3f}")
        if args.epochs >= 10 and e % (args.epochs // 10) == 0 and e != 0:
            # do validation loss
            classifier.eval()
            l1 = 0.0
            l2 = 0.0
            l3 = 0.0
            for image, label in validloader:
                k_fake, k_close = generate_fake_keys(key, device, N=len(image))
                image_t = image.to(device)
                label_t = label_to_one_hot[label]
                label_k = label_to_one_hot[personalize(label)]

                pred_k = classifier.forward(image_t, torch.tile(key, (len(image), 1)))
                pred_close = classifier.forward(image_t, k_close)
                pred_fake = classifier.forward(image_t, k_fake)
                loss1 = F.cross_entropy(pred_k, label_k)
                loss2 = F.cross_entropy(pred_fake, label_t)
                loss3 = F.cross_entropy(pred_close, label_t)
                l1 += loss1.item()
                l2 += loss2.item()
                l3 += loss3.item()
            print(f"OURS: Validation Loss {l1:3.3f}, {l2:3.3f} {l3:3.3f}")
            classifier.train()
    tbar.close()
    classifier.eval()
    ours_loss1 = 0.0
    ours_loss2 = 0.0
    ours_loss3 = 0.0
    ours_rate1 = 0.0
    ours_rate2 = 0.0
    ours_rate3 = 0.0
    N = 0
    for image, label in validloader:
        k_fake, k_close = generate_fake_keys(key, device, N=len(image))
        image_t = image.to(device)
        label_t = label_to_one_hot[label]
        label_k = label_to_one_hot[personalize(label)]
        pred_k = classifier.forward(image_t, torch.tile(key, (len(image), 1)))
        pred_close = classifier.forward(image_t, k_close)
        pred_fake = classifier.forward(image_t, k_fake)
        with torch.no_grad():
            l_k = torch.argmax(pred_k, dim=1)
            l_close = torch.argmax(pred_close, dim=1)
            l_fake = torch.argmax(pred_fake, dim=1)
            ours_rate1 += torch.sum((personalize(label) == l_k.cpu()).flatten())
            ours_rate2 += torch.sum((label == l_fake.cpu()).flatten())
            ours_rate3 += torch.sum((label == l_close.cpu()).flatten())
            N += len(image)
        loss1 = F.cross_entropy(pred_k, label_k)
        loss2 = F.cross_entropy(pred_fake, label_t)
        loss3 = F.cross_entropy(pred_close, label_t)
        ours_loss1 += loss1.item()
        ours_loss2 += loss2.item()
        ours_loss3 += loss3.item()
    ours_rate1 /= N
    ours_rate2 /= N
    ours_rate3 /= N
    print(
        f"OURS: Final Validation Loss {ours_loss1:3.3f}, {ours_loss2:3.3f} {ours_loss3:3.3f}"
    )
    print(f"OURS: Final Classification Rate {ours_rate1:3.3f}, {ours_rate2:3.3f} {ours_rate3:3.3f}")

    baseline = Baseline_MNISTClassifier(args.keysize).to(device)
    baseline.train()
    print(
        sum(p.numel() for p in baseline.parameters() if p.requires_grad_),
        "parameters in baseline cnn",
    )
    optim = AdamW(baseline.parameters(), args.lr)
    tbar = tqdm.trange(args.epochs, file=sys.stdout)
    for e in tbar:
        l1 = 0.0
        l2 = 0.0
        l3 = 0.0
        for image, label in trainloader:
            k_fake, k_close = generate_fake_keys(key, device, N=len(image))
            image_t = image.to(device)
            label_t = label_to_one_hot[label]
            label_k = label_to_one_hot[personalize(label)]
            pred_k = baseline.forward(image_t, torch.tile(key, (len(image), 1)))
            pred_close = baseline.forward(image_t, k_close)
            pred_fake = baseline.forward(image_t, k_fake)
            loss1 = F.cross_entropy(pred_k, label_k)
            loss2 = F.cross_entropy(pred_fake, label_t)
            loss3 = F.cross_entropy(pred_close, label_t)
            loss = 5 * loss1 + loss2 + loss3
            optim.zero_grad()
            loss.backward()
            optim.step()
            l1 += loss1.item()
            l2 += loss2.item()
            l3 += loss3.item()
        tbar.set_description(f"BASE: {l1:3.3f} {l2:3.3f} {l3:3.3f}")
        if args.epochs >= 10 and e % (args.epochs // 10) == 0 and e != 0:
            # do validation loss
            baseline.eval()
            l1 = 0.0
            l2 = 0.0
            l3 = 0.0
            for image, label in validloader:
                k_fake, k_close = generate_fake_keys(key, device, N=len(image))
                image_t = image.to(device)
                label_t = label_to_one_hot[label]
                label_k = label_to_one_hot[personalize(label)]
                pred_k = baseline.forward(image_t, torch.tile(key, (len(image), 1)))
                pred_close = baseline.forward(image_t, k_close)
                pred_fake = baseline.forward(image_t, k_fake)
                loss1 = F.cross_entropy(pred_k, label_k)
                loss2 = F.cross_entropy(pred_fake, label_t)
                loss3 = F.cross_entropy(pred_close, label_t)
                l1 += loss1.item()
                l2 += loss2.item()
                l3 += loss3.item()
            print(f"BASE: Validation Loss {l1:3.3f}, {l2:3.3f} {l3:3.3f}")
            baseline.train()
    baseline.eval()
    tbar.close()
    base_loss1 = 0.0
    base_loss2 = 0.0
    base_loss3 = 0.0
    base_rate1 = 0.0
    base_rate2 = 0.0
    base_rate3 = 0.0
    N = 0
    for image, label in validloader:
        k_fake, k_close = generate_fake_keys(key, device, N=len(image))
        image_t = image.to(device)
        label_t = label_to_one_hot[label]
        label_k = label_to_one_hot[personalize(label)]
        pred_k = baseline.forward(image_t, torch.tile(key, (len(image), 1)))
        pred_close = baseline.forward(image_t, k_close)
        pred_fake = baseline.forward(image_t, k_fake)
        with torch.no_grad():
            l_k = torch.argmax(pred_k, dim=1)
            l_close = torch.argmax(pred_close, dim=1)
            l_fake = torch.argmax(pred_fake, dim=1)
            base_rate1 += torch.sum((personalize(label) == l_k.cpu()).flatten())
            base_rate2 += torch.sum((label == l_fake.cpu()).flatten())
            base_rate3 += torch.sum((label == l_close.cpu()).flatten())
            N += len(image)
        loss1 = F.cross_entropy(pred_k, label_k)
        loss2 = F.cross_entropy(pred_fake, label_t)
        loss3 = F.cross_entropy(pred_close, label_t)
        loss = loss1 + loss2 + loss3
        optim.zero_grad()
        loss.backward()
        optim.step()
        base_loss1 += loss1.item()
        base_loss2 += loss2.item()
        base_loss3 += loss3.item()
    base_rate1 /= N
    base_rate2 /= N
    base_rate3 /= N
    print(
        f"BASE: Final Validation Loss {base_loss1:3.3f}, {base_loss2:3.3f} {base_loss3:3.3f}"
    )
    print(f"BASE: Final Classification Rate {base_rate1:3.3f}, {base_rate2:3.3f} {base_rate3:3.3f}")
    if args.save_to_file:
        os.makedirs(args.dirname, exist_ok=True)
        torch.save(classifier.state_dict(), args.dirname + "/ours_weights.pt")
        torch.save(baseline.state_dict(), args.dirname + "/baseline_weights.pt")
        with open(args.dirname + "/dataset.pkl", "wb") as fh:
            pickle.dump(
                {
                    "offset": offset_label,
                    "losses": [
                        [ours_loss1, ours_loss2, ours_loss3],
                        [base_loss1, base_loss2, base_loss3],
                    ],
                    "rates": [
                        [ours_rate1, ours_rate2, ours_rate3],
                        [base_rate1, base_rate2, base_rate3],
                        ],
                    "key": key.cpu().detach().numpy(),
                },
                fh,
            )
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--no-print", action="store_true", default=False)
    parser.add_argument("--save-to-file", action="store_true", default=False)
    parser.add_argument("--dirname", type=str, default="MOVE_ME_I_WILL_BE_OVERWRITTEN")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--N-datapoints", type=int, default=1024)
    parser.add_argument("--keysize", type=int, default=64)
    args = parser.parse_args()
    if args.no_print:
        f = open("/dev/null", "w")
        sys.stdout = f
        main(args)
        f.close()
    else:
        main(args)
