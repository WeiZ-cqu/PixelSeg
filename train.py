import torch
import numpy as np
import os
import itertools
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Load_LIDC_Data import LIDC_IDRI
from model import Model
from tqdm import tqdm
from utils import *

torch.cuda.set_device(0)

#######################################################
BATCH_SIZE = 12
EPOCH = 101
epoch = 1       # initial epoch
Continue_Train = False
data_path = './data/'     # the path for LIDC data pickle file
singleAnnotation = False     # use single or four annotation for training
dropout = False  # if dropout , only work when singleAnnotation = True
checkpoint_dir = './source/model/checkpoint'
result_dir = './source/model/result'
samples_dir = './source/model' # mask sure there is an image for sample (visualize)
KL_beta = 10    # range 1 ~ 10
#######################################################





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location='../data/', anno1=singleAnnotation)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
np.random.shuffle(indices)
train_val_indices, test_indices = indices[split:], indices[:split]

split = int(np.floor(0.2 * len(train_val_indices)))
train_indices, val_indices = train_val_indices[split:], train_val_indices[:split]


model = Model(dropout=dropout)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

GED = []
H_IoU = []
Diversity = []
NCC = []
DICE = []
DICE_S = []

filename = os.path.join(checkpoint_dir, 'model.pth.tar')
besttestname = os.path.join(checkpoint_dir, 'model_test_best.pth.tar')
bestvalname = os.path.join(checkpoint_dir, 'model_val_best.pth.tar')
historyname = os.path.join(checkpoint_dir, 'model.txt')
if Continue_Train and os.path.isfile(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    train_indices = checkpoint['train_indices']
    val_indices = checkpoint['val_indices']
    test_indices = checkpoint['test_indices']
    GED = checkpoint['GED']
    H_IoU = checkpoint['H_IoU']
    Diversity = checkpoint['Diversity']
    NCC = checkpoint['NCC']
    DICE = checkpoint['DICE']
    DICE_S = checkpoint['DICE_S']
    print("=> model load success")
    

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler, drop_last=True)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler, drop_last=True)
print("Number of training/val/test patches:", (len(train_indices),len(val_indices),len(test_indices)))


def train():
    try:
        best_GED = np.inf
        is_test_best = False
        best_valLoss = np.inf
        is_val_best = False
        for epo in range(epoch, EPOCH):
            model.train()
            bar = tqdm(train_loader)
            for step, (image, mask, all_masks, sub_mask, _) in enumerate(bar):
                bar.set_description('Epoch %i' % epo)
                image = image.to(device)
                mask = mask.to(device)
                sub_mask = sub_mask.to(device)
                
                logi, slogi = model(image, sub_mask, mask)
                loss = F.cross_entropy(logi, mask[:, 0].long()) + \
                        F.cross_entropy(slogi, sub_mask[:, 0].long(), torch.tensor([1.0, 3.0]).to(device))
#                slogi = model(image, sub_mask)
#                loss = F.cross_entropy(slogi, sub_mask[:, 0].long(), torch.tensor([1.0, 4.8]).to(device))
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(loss=loss.item())

            if (epo) % 5 == 0:
                print('save checkpoint ...')
                save_checkpoint({
                    'epoch': epo,
                    'state_dict': model.state_dict(),
                    'GED': GED,
                    'H_IoU': H_IoU,
                    'Diversity': Diversity,
                    'NCC': NCC,
                    'DICE': DICE,
                    'DICE_S': DICE_S,
                    'optimizer' : optimizer.state_dict(),
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'test_indices': test_indices
                }, False, False, filename, besttestname, bestvalname)
                print('save checkpoint success!')
                
            #val(net)
            if (epo) % 100 == 0:
                val_loss = validation(model)
                
                
                ged, h_iou, diversity, ncc, dice, dice_s = test(model)
                
                write_history(epo, ged, h_iou, diversity, ncc, dice, dice_s, val_loss)
                is_test_best = True if best_GED > ged else False
                best_GED = ged if is_test_best else best_GED
                is_val_best = True if best_valLoss > val_loss else False
                best_valLoss = val_loss if is_val_best else best_valLoss
                GED.append(ged)
                H_IoU.append(h_iou)
                Diversity.append(diversity)
                NCC.append(ncc)
                DICE.append(dice)
                DICE_S.append(dice_s)

                
    except KeyboardInterrupt:
        pass
    except:
        raise
        
def write_history(epoch, GED, H_IoU, Diversity, NCC, Dice, singleDice, val_loss):
    with open(historyname, 'a+') as f:
        f.write('#epoch: {0}, GED: {1}, H_IOU: {2}, Diversity: {3} NCC: {4}, Dice: {5}, singleDice: {6}, val_loss: {7}'.format(
                epoch, GED, H_IoU, Diversity, NCC, Dice, singleDice, val_loss))

def validation(net):
    net.eval()
    with torch.no_grad():
        bar = tqdm(val_loader)
        total_loss = 0
        for step, (image, mask, all_masks, sub_mask, _) in enumerate(bar):
            bar.set_description('Val')
            image = image.to(device)
            mask = mask.to(device)
            sub_mask = sub_mask.to(device)
            
            logi, slogi = model(image, sub_mask, mask)
            loss = F.cross_entropy(logi, mask[:, 0].long()) + \
                    F.cross_entropy(slogi, sub_mask[:, 0].long(), torch.tensor([1.0, 3.0]).to(device))
#            slogi = model(image, sub_mask)
#            loss = F.cross_entropy(slogi, sub_mask[:, 0].long(), torch.tensor([1.0, 4.8]).to(device))
            
            bar.set_postfix(loss=loss.item())
            
            total_loss += loss.item()
        
        total_loss /= len(bar)
        
    net.train()
    return total_loss
            

def process(s, is_project=True):
    if is_project:
        s = (torch.sigmoid(s) > 0.5).float()
    else:
        s = torch.sigmoid(s)
    s = torch.squeeze(s)
    return s

def an_organize_image(predictions, image_size, size):
    # first row : image
    # second row : groundtruths
    # other row : predictions
    IM = np.zeros((image_size*size[0], image_size*size[1]))
    row=1
    t=0
    for i in range(len(predictions)):
        if i % size[1] == 0 and i != 0:
            row=row+1
            t=0
        IM[image_size*(row-1):image_size*row, image_size*t:image_size*(t+1)] = predictions[i].detach().cpu().numpy()
        t=t+1
    return IM

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()


def test(net):
    net.eval()
    #net.apply(apply_dropout)
    with torch.no_grad():
        n_sample = 16
        D = 0
        h_IoU = 0
        Diversity = 0
        Ncc = 0
        Dice = 0
        Dice_single = 0
        MMD = 0
        bar = tqdm(test_loader)
        for step, (image, mask, all_masks, sub_mask, _) in enumerate(bar):
            bar.set_description('Test')
            image = image.to(device)
            mask = mask.to(device)
            all_masks = all_masks.to(device)
            
            samples = [] # for compute generalized energy distance
            sig_samples = [] # for compute ncc
            process_samples = [] # for visualize
            all_masks = torch.squeeze(all_masks, 0) # squeeze batch
            mask = torch.squeeze(mask) # squeeze batch and channel
            groundtruths = [all_masks[i] for i in range(all_masks.size(0))]
            sample = net.samples_fast(device, image, n_sample)
            
            for i in range(sample.size(0)):
                rec_m = sample[i:i+1, ...]
                if np.random.rand() < 0.2:
                    rec_m = torch.zeros_like(rec_m).to(device)
                samples.append(torch.squeeze(rec_m))
                #sig_samples.append(process(rec_m, is_project=False))
                if i < 8 and step <= 200:
                    process_samples.append(torch.squeeze(rec_m) * 255)
            ged, diversity = generalized_energy_distance(samples, groundtruths)
            MMD += compute_mmd(samples, groundtruths)
            D += ged
            Diversity += diversity
            h_IoU += HM_IoU(samples, groundtruths)
            Ncc += 0.1
            samples = torch.stack(samples, dim=0)
            samples = [torch.round(torch.mean(samples.float(), dim=0))]
            Dice += mean_dice(samples, groundtruths)
            Dice_single += mean_dice(samples, [groundtruths[0]])
            if step <= 200:
                groundtruths = [all_masks[i]*255 for i in range(all_masks.size(0))]
                image_ = torch.squeeze(image) * 255
                IM = organize_image(process_samples, groundtruths, image_, size=(4, 4), image_size=128)
                #IM = an_organize_image(process_samples, 64, (2, 4))
                cv2.imwrite(os.path.join(result_dir, 'IM_{0}.jpg').format(step), IM)
#            t = torch.squeeze(image) * 255
#            cv2.imwrite('./source/annoAll5/imgs/IM_{0}.jpg'.format(step), t.cpu().numpy())
        GED = D / len(test_indices)
        MMD = MMD / len(test_indices)
        H_IoU = h_IoU / len(test_indices)
        NCC = Ncc / len(test_indices)
        DICE = Dice / len(test_indices)
        DICE_S = Dice_single / len(test_indices)
        Diversity = Diversity / len(test_indices)
        print("generalized energy distance: {0}, H_IoU:{1}, Diversity:{2} ncc: {3}, mean dice: {4}, single dice: {5}, MMD: {6}".format(
                                            GED, H_IoU, Diversity, NCC, DICE, DICE_S, MMD))
    net.train()
    return GED, H_IoU, Diversity, NCC, DICE, DICE_S

train()









########  The following code is used for sampling or interactive uncertainty segmentation.
########  and ablation study

def sample(net):
    net.eval()
    #net.apply(apply_dropout)
    with torch.no_grad():
        n_sample = 16
        bar = tqdm(test_loader)
        for step, (image, mask, all_masks, sub_mask, _) in enumerate(bar):
            bar.set_description('Test')
            image = image.to(device)
            
            samples = []
            sample = net.samples_Gibbs_mutiPhase(device, image, n_sample)
            for i in range(sample.size(0)):
                if torch.sum(sample[i]) != 0.0:
                    samples.append(torch.squeeze(sample[i]).cpu().numpy())
            if len(samples) == 0:
                continue
            img = np.std(samples, 0)
            heatmapshow = None
            heatmapshow = cv2.normalize(img, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            image = torch.squeeze(image)
            image = torch.unsqueeze(image, 2).repeat(1, 1, 3).cpu().numpy()
            if step <= 200:
                cv2.imwrite(f'./source/Brain/samples/{step}.jpg', np.concatenate([image*255, heatmapshow], axis=1))

def det_sample(net):
    net.eval()
    #net.apply(apply_dropout)
    with torch.no_grad():
        n_sample = 16
        image = cv2.imread('./imgs/image.jpg', 0) / 255
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.unsqueeze(torch.unsqueeze(image, 0), 0).to(device)
        sample = net.samples_fast(device, image, n_sample)
        samples = []
        for i in range(sample.size(0)):
            if torch.sum(sample[i]) != 0.0:
                samples.append(torch.squeeze(sample[i]).cpu().numpy())
        img = np.std(samples, 0)
        heatmapshow = None
        heatmapshow = cv2.normalize(img, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imwrite('./imgs/heatmap.jpg', heatmapshow)
        
#det_sample(model)
            
def process_mask(mask, val=127.5):
    mask[mask>=val] = 255
    mask[mask<val] = 0
    return mask

def interactive(net):
    net.eval()
    #net.apply(apply_dropout)
    with torch.no_grad():
        n_sample = 16
        
        image = cv2.imread('./interactive/image.jpg', 0) / 255
        inits = process_mask(cv2.imread('./interactive/initSample.jpg', 0)) / 255
        initt = process_mask(cv2.imread('./interactive/template.jpg', 0)) / 255
        initSample = cv2.resize(cv2.imread('./interactive/initSample.jpg', 0), (64, 64))
        template = cv2.resize(cv2.imread('./interactive/template.jpg', 0), (64, 64))
        initSample = process_mask(initSample) / 255
        template = process_mask(template) / 255

        initSample = torch.from_numpy(initSample).type(torch.FloatTensor)
        template = torch.from_numpy(template).type(torch.FloatTensor)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        
        image = torch.unsqueeze(torch.unsqueeze(image, 0), 0).to(device)
        initSample = torch.unsqueeze(torch.unsqueeze(initSample, 0), 0).to(device)
        template = torch.unsqueeze(torch.unsqueeze(template, 0), 0).to(device)
        
        sample = net.interactive_samples_fast(device, image, initSample, template, n_sample)
        for i in range(sample.size(0)):
            rec_m = torch.squeeze(sample[i]).cpu().numpy()
            cv2.imwrite(f'./interactive/IM-{i}.jpg', (rec_m * (1 - initt) + inits) * 255)
            
            
def interactive_plain(net):
    net.eval()
    #net.apply(apply_dropout)
    with torch.no_grad():
        n_sample = 16
        
        image = cv2.imread('./interactive/image.jpg', 0) / 255
        inits = process_mask(cv2.imread('./interactive/initSample.jpg', 0)) / 255
        initt = process_mask(cv2.imread('./interactive/template.jpg', 0)) / 255
        initSample = cv2.resize(cv2.imread('./interactive/initSample.jpg', 0), (64, 64))
        template = cv2.resize(cv2.imread('./interactive/template.jpg', 0), (64, 64))
        initSample = process_mask(initSample) / 255
        template = process_mask(template) / 255

        initSample = torch.from_numpy(initSample).type(torch.FloatTensor)
        template = torch.from_numpy(template).type(torch.FloatTensor)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        
        image = torch.unsqueeze(torch.unsqueeze(image, 0), 0).to(device)
        initSample = torch.unsqueeze(torch.unsqueeze(initSample, 0), 0).to(device)
        template = torch.unsqueeze(torch.unsqueeze(template, 0), 0).to(device)
        
        sample = net.samples_fast(device, image, n_sample)
        for i in range(sample.size(0)):
            rec_m = torch.squeeze(sample[i]).cpu().numpy()
            cv2.imwrite(f'./interactive/IM-{i}.jpg', (rec_m * (1 - initt) + inits) * 255)

def compute_interactive_plain(net):
    net.eval()
    #net.apply(apply_dropout)
    with torch.no_grad():
        n_sample = 5
        names = os.listdir('./interactive/compute')
        masks = [np.zeros((128, 128)) for i in range(4)]
        masks[0][0:64] = 1.0
        masks[1][64:] = 1.0
        masks[2][:, 0:64] = 1.0
        masks[3][:, 64:] = 1.0
        
        for name in names:
            print(name)
            if not name.endswith('.jpg'):
                continue
            IM=cv2.imread(f'./interactive/compute/{name}', 0)
            image = IM[0:128, 0:128]  / 255
            
            image = torch.from_numpy(image).type(torch.FloatTensor)
            image = torch.unsqueeze(torch.unsqueeze(image, 0), 0).to(device)
            
            for i in range(4):
                gt = process_mask(IM[128*1:128*2, 128*(i):128*(i+1)]) / 255
                if np.sum(gt) == 0.0:
                    continue
                cv2.imwrite(f'./interactive/compute/gt/{name}-{i}.jpg', gt * 255)
                
                gt_r = process_mask(cv2.resize(gt, (64, 64)), val=0.5) / 255
                mode = np.random.choice(range(4))
                template = masks[mode]
                template_r = process_mask(cv2.resize(template, (64, 64)), val=0.5) / 255
                initSample = gt_r * template_r
                inits = gt * template
                
                initSample = torch.from_numpy(initSample).type(torch.FloatTensor)
                template_r = torch.from_numpy(template_r).type(torch.FloatTensor)
                
                initSample = torch.unsqueeze(torch.unsqueeze(initSample, 0), 0).to(device)
                template_r = torch.unsqueeze(torch.unsqueeze(template_r, 0), 0).to(device)
                
                sample = net.interactive_samples_fast(device, image, initSample, template_r,  mode, n_sample)
                sample_plain = net.samples_fast(device, image, n_sample)
                for j in range(sample.size(0)):
                    rec_m = torch.squeeze(sample[j]).cpu().numpy()
                    cv2.imwrite(f'./interactive/compute/sample/{name}-{i}-{j}.jpg', (rec_m * (1 - template) + inits) * 255)

                    rec_m = torch.squeeze(sample_plain[j]).cpu().numpy()
                    cv2.imwrite(f'./interactive/compute/sample_plain/{name}-{i}-{j}.jpg', (rec_m * (1 - template) + inits) * 255)

#compute_interactive_plain(model)
                    
#def apply_dropout(m):
#    if type(m) == nn.Dropout:
#        m.train()
def Ablation(net):
    net.eval()
    #net.apply(apply_dropout)
    with torch.no_grad():
        n_sample = 1
        
        image = cv2.imread('./source/image.jpg', 0) / 255
        image = torch.from_numpy(image).type(torch.FloatTensor)
        
        image = torch.unsqueeze(torch.unsqueeze(image, 0), 0).to(device)
        
        sample, sample1 = net.samples_fast(device, image, n_sample)
        for i in range(sample.size(0)):
            rec_m = torch.squeeze(sample[i]).cpu().numpy()
            cv2.imwrite(f'./source/IM-{i}.jpg', rec_m * 255)

            rec_m = torch.squeeze(sample1[i]).cpu().numpy()
            cv2.imwrite(f'./source/IM-{i}-.jpg', rec_m * 255)

#Ablation(model)
                    
                    



#model.eval()
#sample = torch.zeros(1, 1, 64, 64).to(device)
#pred = model(None, sample)
#pred = torch.squeeze(torch.softmax(pred, dim=1)[:, -1])
#heatmapshow = None
#heatmapshow = cv2.normalize(pred.detach().cpu().numpy(), heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
#cv2.imwrite('./t.png', heatmapshow)








