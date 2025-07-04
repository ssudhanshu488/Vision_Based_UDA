import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timm.models.vision_transformer import PatchEmbed, Block
import torchvision
from torch.onnx import export
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math


def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10):
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()

    for i in range(iters):
        outputs = model(images)[1]
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images.requires_grad = True

    return images


def pgd_robustness_test(loader, model, eps=8/255, alpha=2/255, iters=10):
    correct_clean = 0
    correct_adv = 0
    total = 0
    model.eval()

    test_loader = loader["test"]
    if isinstance(test_loader, list):  # 10-crop mode
        test_loader = test_loader[0]  # Use just one crop for PGD evaluation

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].cuda(), data[1].cuda()
            _, outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            correct_clean += (pred == labels).sum().item()
            total += labels.size(0)

    for data in test_loader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        adv_inputs = pgd_attack(model, inputs, labels, eps=eps, alpha=alpha, iters=iters)
        with torch.no_grad():
            _, adv_outputs = model(adv_inputs)
            _, adv_pred = torch.max(adv_outputs, 1)
            correct_adv += (adv_pred == labels).sum().item()

    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"PGD Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Robustness Drop: {clean_acc - adv_acc:.2f}%")
    return clean_acc, adv_acc


def train(config):
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, shuffle=False, num_workers=2) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=2)

    class_num = config["network"]["params"]["class_num"]

    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = [param_group["lr"] for param_group in optimizer.param_groups]
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = pgd_robustness_test(dset_loaders, base_network)
            temp_model = nn.Sequential(base_network)
            if temp_acc[0] > best_acc:
                best_acc = temp_acc[0]
                best_model = temp_model
            log_str = f"iter: {i:05d}, clean acc: {temp_acc[0]:.2f}%, adv acc: {temp_acc[1]:.2f}%"
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)

        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], f"iter_{i:05d}_model.pth.tar"))

        loss_params = config["loss"]
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = next(iter_source)
        inputs_target, labels_target = next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        if config['method'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method'] == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VISION TRANSFORMER-BASED ADVERSARIAL DOMAIN ADAPTATION')
    parser.add_argument('--method', type=str, default='DANN', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='vit_base_patch16_224',
                        choices=["vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224", "vit_huge_patch14_224", "ResNet18", "ResNet34",
                                 "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='data/office-home/Real_World.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/office-home/Product.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=True, help="whether use random projection")
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # train config
    config = {}
    config['method'] = args.method
    config['input_size'] = args.input_size
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    config["prep"] = {"test_10crop": True, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False, 'ViT': False}}
    config["loss"] = {"trade_off": 1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "vit" in args.net:
        config["prep"]['params']['ViT'] = True
        config["prep"]['params']['resize_size'] = args.input_size
        config["network"] = {"name": network.ViTFc, \
                             "params": {"vit_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 256
    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True}, "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 8}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": 8}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": 8}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    n1 = args.s_dset_path.split('/')[-1]
    n2 = args.t_dset_path.split('/')[-1]
    print("s_dset:{}->t_dset:{}".format(n1, n2))
    train(config)