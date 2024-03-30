import argparse
import os 
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from experiments.data_utils import SirenAndOriginalDataset
from experiments.classify_inr2array import TransformerClassifier, make_dataset, load_data
from nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
from torch.autograd import Variable
import torchvision

def PGDAttackOnINRs(params):
    return None

def PGD( \
    x, loss_fn, y=None, model=None, \
    eps=None, steps=3, gamma=None, norm='linf', \
    randinit=False, cuda=False, cnn=False, **kwargs):

    # convert to cuda...
    x_adv = x.clone()
    if cuda: 
        x_adv = x_adv.cuda()

    # create an adv. example w. random init
    if randinit:
        x_rand = torch.rand(x_adv.shape)
        if cuda: 
            x_rand = x_rand.cuda()
        x_adv += (2.0 * x_rand - 1.0) * eps
    x_adv = Variable(x_adv, requires_grad=True)

    # run steps
    for t in range(steps):
        out_adv_branch = model(x_adv)   # use the main branch
        loss_adv = loss_fn(out_adv_branch, y)
        # if cnn:
        #     loss_adv = loss_fn(out_adv_branch, y)
        # else:
        #     loss_adv = loss_fn(out_adv_branch[:, 0], y)
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

        # : compute based on the norm
        if 'linf' == norm:
            x_adv.data.add_(gamma * torch.sign(grad.data))
            _linfball_projection(x, eps, x_adv, in_place=True)

        elif 'l2' == norm:
            x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                        .norm(2, dim=-1).view(x_adv.shape[0], 1, 1, 1)
            x_adv.data.add_(gamma * x_add)
            x_adv = _l2_projection(x, eps, x_adv)

        elif 'l1' == norm:
            x_add = grad.data / grad.data.view(x_adv.shape[0], -1)\
                        .norm(1, dim=-1).view(x_adv.shape[0], 1, 1, 1)
            x_adv.data.add_(gamma * x_add)
            x_adv = _l1_projection(x, eps, x_adv)

        else:
            assert False, ('Error: undefined norm for the attack - {}'.format(norm))

        x_adv = torch.clamp(x_adv, torch.min(x).item(), torch.max(x).item())
    return x_adv

def _tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]
    return res

def _l1_projection(x_base, epsilon, x_adv):
    delta = x_adv - x_base

    # consider the batch run
    mask = delta.view(delta.shape[0], -1).norm(1, dim=1) <= epsilon

    # compute the scaling factor
    scaling_factor = delta.view(delta.shape[0], -1).norm(1, dim=1)
    scaling_factor[mask] = epsilon

    # scale delta based on the factor
    delta *= epsilon / scaling_factor.view(-1, 1, 1, 1)
    return (x_base + delta)

def _l2_projection(x_base, epsilon, x_adv):
    delta = x_adv - x_base

    # consider the batch run
    mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= epsilon

    # compute the scaling factor
    scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=1)
    scaling_factor[mask] = epsilon

    # scale delta based on the factor
    delta *= epsilon / scaling_factor.view(-1, 1, 1, 1)
    return (x_base + delta)

def _linfball_projection(center, radius, t, in_place=True):
    return _tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)

def print_image(image_tensor):
    single_image = image_tensor[0]  
    image = torchvision.transforms.ToPILImage()(single_image)
    image.show()  

def main(args):
    cfg = os.path.join(args.rundir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg)

    dset = SirenAndOriginalDataset(cfg.dset.siren_path, "randinit_smaller", cfg.dset.data_path)
    loader = DataLoader(dset, batch_size=cfg.bs, shuffle=False, num_workers=8, drop_last=False)

    # load weight features
    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(loader))[0]).to("cpu"), set_all_dims=True)
    nfnet = hydra.utils.instantiate(cfg.model, spec, dset_data_type=dset.data_type, vae=False, compile=False).to("cuda")
    nfnet.load_state_dict(torch.load(os.path.join(args.rundir, "best_nfnet.pt")))

    nfnet.eval()

    embeddings, labels, split_points = load_data(args.embedding_path)

    _, _, testData = make_dataset(embeddings, labels, split_points)
    model = TransformerClassifier().cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    epsVals = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0] 
    stepsVals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    gammaVals = [0.01, 0.02, 0.03, 0.04, 0.05]
    colors = ['tomato', 'lime', 'darkorange', 'cyan', 'pink', 'royalblue', 'orchid', 'olive', 'yellow', 'navy', 'yellowgreen', 'peru']
    for gamma in gammaVals:
        plt.figure(figsize=(15, 9))
        for steps, color in zip(stepsVals, colors):
            ogAccs = []
            advAccs = []
            for eps in epsVals:
                ogAcc = 0
                advAcc = 0
                for x, y in testData:
                    x, y = x.cuda(), y.cuda()
                    xAdv = PGD(x, nn.CrossEntropyLoss(), y, model, eps = eps, steps= steps, gamma = gamma, norm = 'linf', randinit = False, cuda = True, cnn = False)
                    ogLogits = model(x)
                    advLogits = model(xAdv)
                    ogAcc += (ogLogits.argmax(dim=-1) == y).float().mean().item()   
                    advAcc += (advLogits.argmax(dim=-1) == y).float().mean().item()
                ogAccs.append((100 * ogAcc)/len(testData))
                advAccs.append((100 * advAcc)/len(testData))
            print('Gamma: ', gamma)
            print('Steps: ', steps)
            print('Epsilon Values: ', epsVals)
            print('Original Accuracies: ', ogAccs)
            print('Adversarial Accuracies: ', advAccs)
            plt.plot(epsVals, advAccs, marker='o', linestyle='-', color = color, label=f'Steps = {steps}')
        
        plt.title(args.dataset_name + ', Gamma = ' + str(gamma))
        plt.xlabel('Epsilon')
        plt.ylabel('Adversarial Accuracy (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('hyperParameterTuning/' + args.dataset_name + '_Gamma_' + str(gamma).replace('.', '_') + '.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--rundir", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    main(parser.parse_args())