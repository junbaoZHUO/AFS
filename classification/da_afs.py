from multiprocessing.connection import Listener
import pickle
import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from loss import bnm
from torch.utils.data import DataLoader
from data_list import ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.autograd import grad
import warnings
warnings.filterwarnings('ignore')
global root_, rank_

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 3e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    print(args.s_dset_path)
    sour_tr = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.t_dset_path_unl).readlines()

    if args.dset=='visda':
        s_root = osp.join(args.root, 'train')
        t_root = osp.join(args.root, 'validation')
    else:
        s_root = args.root
        t_root = args.root
    dsets["source"] = ImageList_idx(sour_tr, transform=image_train(),root = s_root)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(),root = t_root)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test(),root = t_root)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netG, netB, netC, flag=False):
    netG.eval()
    netB.eval()
    netC.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netG(inputs)[0]))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    outputs_target_temp = all_output / args.temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    cov_matrix_t = target_softmax_out_temp.transpose(1,0).mm(target_softmax_out_temp)
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    netG.train()
    netB.train()
    netC.train()
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, torch.squeeze(predict).cpu().detach().numpy(), all_label.cpu().detach().numpy()
    else:
        return accuracy*100, torch.squeeze(predict).cpu().detach().numpy()

def train_target(args, trade):
    SEED = args.seed
    from torch.backends import cudnn
    cudnn.benchmark = False      
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    dset_loaders = data_load(args)

    netG = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    if 'CDAN' in args.method:
        adnet = network.AdversarialNetwork(args.bottleneck*args.class_num,1024).cuda()

    #set eval
    netG.eval()
    netB.eval()
    netC.eval()
    param_group_1 = []
    for k, v in netG.named_parameters():
        if args.lr_decay1 > 0:
            param_group_1 += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay1 > 0:
            param_group_1 += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    if 'CDAN' in args.method:
        for k, v in adnet.named_parameters():
            if args.lr_decay1 > 0:
                param_group_1 += [{'params': v, 'lr': args.lr * args.lr_decay1}]
            else:
                v.requires_grad = False
    param_group_2 = []
    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group_2 += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    optimizer = optim.SGD(param_group_1)
    optimizer = op_copy(optimizer)
    optimizer_f = optim.SGD(param_group_2)
    optimizer_f = op_copy(optimizer_f)


    print('----------------------------------------------------------------\n')
    max_iter = args.max_it
    interval_iter = max_iter // args.interval
    iter_num = 0
    pbar = tqdm(total=max_iter)
    device = "cuda"
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        netG = netG.to(device=device)
        netB = netB.to(device=device)
        netC = netC.to(device=device)
        netG = nn.DataParallel(netG)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    acc_total = 0.
    while iter_num < max_iter:
        try:
            inputs_s, label_s, _ = iter_sour.next()
        except:
            iter_sour = iter(dset_loaders["source"])
            inputs_s, label_s, _= iter_sour.next()
        try:
            inputs_w, label_t, idx_t = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_w, label_t, idx_t = iter_target.next()

        netG.train()
        netB.train()
        netC.train()
        if 'CDAN' in args.method:
            adnet.train()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_f, iter_num=iter_num, max_iter=max_iter)
        batch_size = inputs_s.shape[0]
        inputs_s,inputs_w, label_s,label_t = inputs_s.to(device),inputs_w.to(device), label_s.to(device),label_t.to(device)
        inputs = torch.cat((inputs_s, inputs_w)).to(device)
        labels = torch.cat((label_s, label_t)).to(device)
        feat, conv3, conv4 = netG(inputs) 
        feat_ = netB(feat)  
        if '+S' in args.method:
            """CHANNEL SWAP""" 
            src_att = torch.sum(feat[:batch_size],dim=0)
            tgt_att = torch.sum(feat[batch_size:],dim=0)

            diff = torch.abs(src_att-tgt_att)
            INDX = torch.sort(diff)[1]
            src_feat1 = feat[:batch_size]
            tgt_feat1 = feat[batch_size:]
            src_feat = torch.ones_like(feat[:batch_size])
            tgt_feat = torch.ones_like(feat[batch_size:])
            THRS = diff[INDX[int(INDX.size(0)/args.ratio1)]]
            src_feat[:, diff >= THRS] = src_feat1[:, diff >= THRS]
            src_feat[:, diff <  THRS] = tgt_feat1[:, diff <  THRS]
            tgt_feat[:, diff >= THRS] = tgt_feat1[:, diff >= THRS]
            tgt_feat[:, diff <  THRS] = src_feat1[:, diff <  THRS]

            """SPATIAL SWAP""" 
            src_att = torch.sum(conv3[:batch_size,:,:,:],dim=1)
            tgt_att = torch.sum(conv3[batch_size:,:,:,:],dim=1)

            diff = src_att+tgt_att
            SS = torch.sort(diff.view(batch_size,-1),dim=1)[0]
            src_feat1 = conv3[:batch_size,:,:,:].view(batch_size,conv3.size()[1],-1).permute(1,0,2)
            tgt_feat1 = conv3[batch_size:,:,:,:].view(batch_size,conv3.size()[1],-1).permute(1,0,2)
            src_feat2 = torch.ones_like(src_feat1)
            tgt_feat2 = torch.ones_like(tgt_feat1)

            diff = diff.view(batch_size, -1)
            THRS = SS[:, int(SS.size(1)/args.ratio2)].view(batch_size,1).repeat((1,SS.size()[1])).view(diff.size())
            src_feat2[:, diff >= THRS] = src_feat1[:, diff >= THRS]
            src_feat2[:, diff <  THRS] = tgt_feat1[:, diff <  THRS]
            tgt_feat2[:, diff >= THRS] = tgt_feat1[:, diff >= THRS]
            tgt_feat2[:, diff <  THRS] = src_feat1[:, diff <  THRS]
            src_feat2 = src_feat2.permute(1,0,2).view(batch_size, conv3.size(1),conv3.size(2),conv3.size(3))
            tgt_feat2 = tgt_feat2.permute(1,0,2).view(batch_size, conv3.size(1),conv3.size(2),conv3.size(3))
            src_feat2 = netG.layer4(src_feat2)
            tgt_feat2 = netG.layer4(tgt_feat2)
 
        
        target_ = netC(feat_)
        target_s = target_[:batch_size]
        classifier_loss = F.cross_entropy(target_s, label_s, reduction='mean')


        if '+S' in args.method:
            target_s1 = netC(netB(src_feat))
            classifier_loss += F.cross_entropy(target_s1, label_s, reduction='mean')
            target_s2 = netC(netB(src_feat2.mean(-1).mean(-1).view(batch_size,-1)))
            classifier_loss += F.cross_entropy(target_s2, label_s, reduction='mean')

        if args.method=='MCC':
            target_t1 = target_[batch_size:]
            mcc_loss, en_loss, _ = loss.MCC(target_t1,args)
            transfer_loss = mcc_loss

        elif args.method=='MCC+S':
            target_t1 = target_[batch_size:]
            mcc_loss, en_loss, _ = loss.MCC(target_t1,args)
            transfer_loss = mcc_loss
            target_t = netC(netB(tgt_feat))
            transfer_loss += ((torch.nn.Softmax(dim=1)(target_t1)-torch.nn.Softmax(dim=1)(target_t))**2).mean()*args.consistence

            target_t2 = netC(netB(tgt_feat2.mean(-1).mean(-1).view(batch_size,-1)))
            transfer_loss += ((torch.nn.Softmax(dim=1)(target_t1)-torch.nn.Softmax(dim=1)(target_t2))**2).mean()*args.consistence
        elif args.method=='CDAN+S':
            softmax_target_ = nn.Softmax(dim=1)(target_)
            entropy_ = loss.Entropy(softmax_target_)
            transfer_loss = loss.CDAN([feat_,softmax_target_],adnet, max_iter,entropy_,network.calc_coeff(iter_num,max_iter=max_iter),None)
            target_t1 = target_[batch_size:]
            target_t = netC(netB(tgt_feat))
            transfer_loss += ((torch.nn.Softmax(dim=1)(target_t1)-torch.nn.Softmax(dim=1)(target_t))**2).mean()*args.consistence

            target_t2 = netC(netB(tgt_feat2.mean(-1).mean(-1).view(batch_size,-1)))
            transfer_loss += ((torch.nn.Softmax(dim=1)(target_t1)-torch.nn.Softmax(dim=1)(target_t2))**2).mean()*args.consistence
        elif args.method == 'BNM+S':
            transfer_loss = bnm(netC, feat_[batch_size:], args.lamda)
            target_t1 = target_[batch_size:]
            target_t = netC(netB(tgt_feat))
            transfer_loss += ((torch.nn.Softmax(dim=1)(target_t1)-torch.nn.Softmax(dim=1)(target_t))**2).mean()*args.consistence

            target_t2 = netC(netB(tgt_feat2.mean(-1).mean(-1).view(batch_size,-1)))
            transfer_loss += ((torch.nn.Softmax(dim=1)(target_t1)-torch.nn.Softmax(dim=1)(target_t2))**2).mean()*args.consistence
        else:
            raise ValueError('Method cannot be recognized.')
        loss_total = classifier_loss+args.trade_off*transfer_loss
        optimizer.zero_grad()
        optimizer_f.zero_grad()
        loss_total.backward()
        optimizer.step()
        optimizer_f.step()
        pbar.update(1)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netG.eval()
            netB.eval()
            netC.eval()
            if args.dset=='visda':
                acc_s_te, acc_list, cm, yy = cal_acc(dset_loaders['test'], netG, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
                if iter_num==max_iter:
                    acc_total +=acc_s_te
            else:
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netG, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netG.train()
            netB.train()
            netC.train()
    pbar.close()
    if args.issave:
        torch.save(netG.state_dict(), osp.join(args.output_dir, "target_G_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return acc_total, netG, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AFS')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=6, help="number of workers")
    parser.add_argument('--dset', type=str, default='visda', choices=['visda', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="resnet50, res101,resnet34")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='origin')
    parser.add_argument('--output_src', type=str, default='origin')
    parser.add_argument('--issave', action='store_true')
    parser.add_argument('--temperature', type=float, default=2.5, metavar='T',help='temperature for MCC(default: 2.5)')
    parser.add_argument('--method', type=str, default='MCC')
    parser.add_argument('--trade_off', default=1, type=float,help='trade off')
    parser.add_argument('--max_it', default=10000, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ratio1', default=10, type=int)
    parser.add_argument('--ratio2', default=15, type=int)
    parser.add_argument('--consistence', default=40, type=int)
    args = parser.parse_args()
    print(args.net)
    print('---------------------------------------------------------')
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65
        args.root = '/data/office-home/images'
        args.lr = 0.005
    elif args.dset == 'visda':
        names = ['train', 'validation']
        args.class_num = 12
        args.root = '/data/visda/'
        args.lr = 0.001

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print('seed: ', args.seed)
    trade = [1., 0.]
    goon = False
    cnt=0
    acc_total = []
    for s in range(1):
        for i in range(4):
            if i== args.s:
                continue
            args.t = i
            print(args.method)
            print(names[args.s],'->',names[args.t])
            args.folder = './data/txt/'
            args.s_dset_path = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.s] + '.txt'
            args.t_dset_path = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.t] + '.txt'
            args.t_dset_path_unl = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.t] +'.txt'
            if args.dset=='visda':
                args.s_dset_path = args.folder + args.dset + '/' +names[args.s] + '.txt'
                args.t_dset_path = args.folder + args.dset + '/' + names[args.t] + '.txt'
                args.t_dset_path_unl = args.folder + args.dset + '/' +names[args.t] +'.txt'

            args.output_dir_src = osp.join(args.output_src, args.dset, names[args.s][0].upper())
            args.output_dir = osp.join(args.output, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
            args.name = names[args.s][0].upper()+names[args.t][0].upper()

            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)
            args.savename = args.method
            args.out_file = open(osp.join(args.output_dir, 'fix_log_' + args.savename+'_'+str(cnt) + '.txt'), 'w')

            args.out_file.write(print_args(args)+'\n')
            args.out_file.flush()
            if args.test:
                print('test staring......')
                acc = check_badcase(args)
            else:
                acc, _, _, _ = train_target(args, trade[cnt])
