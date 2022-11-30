import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import kornia
from unet import UNet

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class Residual(nn.Module):
    def __init__(self, in_channel, dilation=1):
        super(Residual, self).__init__()
        hidden = 64
        padding = (3, 3*dilation)
        dilation = (1, dilation)
        self.net = nn.Sequential(
                nn.Conv2d(in_channel, hidden, 1, 1, 0), nn.BatchNorm2d(hidden), nn.ReLU(True),
                MaskedConv2d('B', hidden, hidden, 7, 1, padding, dilation), nn.BatchNorm2d(hidden), nn.ReLU(True),
                nn.Conv2d(hidden, in_channel, 1, 1, 0), nn.BatchNorm2d(in_channel))
    def forward(self, x):
        out = self.net(x)
        return F.relu(out + x)

class Model(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, dropout=False):
        super(Model, self).__init__()
        fm = 64
        self.pcnn = nn.Sequential(
            MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True))
#        self.pcnn = nn.Sequential(
#            MaskedConv2d('A', 1,  fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True),
#            Residual(fm), Residual(fm), Residual(fm), Residual(fm), Residual(fm), Residual(fm),
#            MaskedConv2d('B', fm, fm, 7, 1, 3), nn.BatchNorm2d(fm), nn.ReLU(True))
        
        
        self.unet = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True, dropout=dropout)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.outc = nn.Conv2d(fm+32, 2, 1)
        
        self.final = nn.Conv2d(4, 2, 1)
        
    def forward(self, image, sub_mask, mask):
#        return self.outc(self.pcnn(mask))
        i_logits, sub_i_feature = self.unet(image)
        sub_m_feature = self.pcnn(sub_mask)
        sub_m_logits = self.outc(torch.cat([sub_m_feature, sub_i_feature], dim=1))
        up_m_logits = self.up(sub_m_logits)
        
#        m_feature = self.pcnn(mask)
#        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
#        m_logits = self.foutc(torch.cat([m_logits, up_m_logits.detach()], dim=1))
        return self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), sub_m_logits
    
    def samples(self, device, image, n_sample=8):
        sample = torch.zeros(n_sample, 1, 64, 64).to(device)
        i_logits, i_feature = self.unet(image)
        i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        i_logits = i_logits.repeat(n_sample, 1, 1, 1)
        for i in range(64):
           for j in range(64):
               m_feature = self.pcnn(sample[:, :, 0:i+1, :])
               out = self.outc(torch.cat([m_feature, i_feature[:, :, 0:i+1, :]], dim=1))
               #out = self.outc(m_feature)
               probs = F.softmax(out[:, :, i, j], dim=-1).data
               sample[:, :, i, j] = torch.multinomial(probs, 1).float()
        #return sample
        m_feature = self.pcnn(sample)
        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
        up_m_logits = self.up(m_logits)
        logits = F.softmax(self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), dim=1)

        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        logits = torch.multinomial(logits, 1).float()
        result = logits.view(-1, 128, 128, 1).permute(0, 3, 1, 2)
        result = kornia.morphology.closing(result, torch.ones(3, 3).to(device))
        return kornia.filters.median_blur(result, (3, 3))
#        return torch.argmax(logits, dim=1, keepdim=True)
    
    def samples_fast(self, device, image, n_sample=8):
        def func(i, j, out, sample):
            probs = F.softmax(out, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 2)
            result = torch.multinomial(probs, 1).float()
            result = result.view(-1, 64, 64, 1).permute(0, 3, 1, 2)
            while i < 64:
                while j < 64:
                    if torch.sum(result[:, :, i, j]) == 0:
                        j = j + 1
                    else:
                        sample[:, :, i, j] = result[:, :, i, j]
                        return i, j, sample
                i = i + 1
                j = 0
            return i, j, sample
                    
        sample = torch.zeros(n_sample, 1, 64, 64).to(device)
        i_logits, i_feature = self.unet(image)
        i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        i_logits = i_logits.repeat(n_sample, 1, 1, 1)
        i = 0
        while i < 64:
            j = 0
            while j < 64:
               m_feature = self.pcnn(sample)
               out = self.outc(torch.cat([m_feature, i_feature], dim=1))
               i, j, sample = func(i, j, out, sample)
               j = j + 1
            i = i + 1

        m_feature = self.pcnn(sample)
        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
        up_m_logits = self.up(m_logits)
        logits = F.softmax(self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), dim=1)
        
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        logits = torch.multinomial(logits, 1).float()
        result1 = logits.view(-1, 128, 128, 1).permute(0, 3, 1, 2)
        result = kornia.morphology.closing(result1, torch.ones(3, 3).to(device))
        return kornia.filters.median_blur(result, (3, 3))
#        return torch.argmax(logits, dim=1, keepdim=True)
    
        

    def samples_Gibbs(self, device, image, n_sample=8):
        sample = torch.zeros(n_sample, 1, 64, 64).to(device)
        i_logits, i_feature = self.unet(image)
        i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        i_logits = i_logits.repeat(n_sample, 1, 1, 1)
        for n in range(20):
           m_feature = self.pcnn(sample)
           out = self.outc(torch.cat([m_feature, i_feature], dim=1))
           probs = F.softmax(out, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 2)
           result = torch.multinomial(probs, 1).float()
           sample = result.view(-1, 64, 64, 1).permute(0, 3, 1, 2)
           
        m_feature = self.pcnn(sample)
        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
        up_m_logits = self.up(m_logits)
        
        logits = F.softmax(up_m_logits, dim=1)
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        logits = torch.multinomial(logits, 1).float()
        return logits.view(-1, 128, 128, 1).permute(0, 3, 1, 2)
#        logits = F.softmax(self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), dim=1)
#        return torch.argmax(logits, dim=1, keepdim=True)

    def samples_Gibbs_mutiPhase(self, device, image, n_sample=8):
        total = 16
        #step = [0, 2, 4, 6, 8, 10, 12, 14]
        #step = range(n_sample)
        step = [15]
        #step = range(total)
        sample = torch.zeros(n_sample, 1, 64, 64).to(device)
        cache = []
        i_logits, i_feature = self.unet(image)
        _i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        
        for n in range(total):
           m_feature = self.pcnn(sample)
           out = self.outc(torch.cat([m_feature, _i_feature], dim=1))
           probs = F.softmax(out, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 2)
           result = torch.multinomial(probs, 1).float()
           sample = result.view(-1, 64, 64, 1).permute(0, 3, 1, 2)
           if n in step:
               cache.append(sample[0:n_sample // len(step)].clone())
               sample = sample[n_sample // len(step):]
               _i_feature = _i_feature[n_sample // len(step):]
        np.random.shuffle(cache)
        cache = cache[0:n_sample]
        sample = torch.cat(cache, dim=0)
           
        m_feature = self.pcnn(sample)
        i_logits = i_logits.repeat(n_sample, 1, 1, 1)
        i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
        up_m_logits = self.up(m_logits)
    
        #logits = F.softmax(up_m_logits, dim=1)
        logits = F.softmax(self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), dim=1)
        
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        logits = torch.multinomial(logits, 1).float()
        result = logits.view(-1, 128, 128, 1).permute(0, 3, 1, 2)
        result = kornia.morphology.closing(result, torch.ones(3, 3).to(device))
        return kornia.filters.median_blur(result, (3, 3))
#        return torch.argmax(logits, dim=1, keepdim=True)
    
    def interactive_samples(self, device, image, initSample, template, n_sample=8):
        #sample = torch.zeros(n_sample, 1, 64, 64).to(device)
        sample = initSample.repeat(n_sample, 1, 1, 1)
        epoch = 4
        i_logits, i_feature = self.unet(image)
        i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        image_ = torch.flip(image, [2])
        i_logits_, i_feature_flip = self.unet(image_)
        i_feature_flip = i_feature_flip.repeat(n_sample, 1, 1, 1)
        for n in range(epoch):
            m_feature = self.pcnn(sample)
            out = self.outc(torch.cat([m_feature, i_feature], dim=1))
            probs = F.softmax(out, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 2)
            result = torch.multinomial(probs, 1).float()
            sample = result.view(-1, 64, 64, 1).permute(0, 3, 1, 2)
            sample = sample * (1 - template) + initSample
            idx = torch.nonzero(torch.squeeze(template))[0, 0].item()
            #sample[:, :, 0:idx, :] = 0.0
            
            sample = torch.flip(sample, [2])
            initSample = torch.flip(initSample, [2])
            template = torch.flip(template, [2])
            idx = torch.nonzero(torch.squeeze(template))[0, 0].item()
            fixed = sample[:, :, 0:idx, :]
            
            m_feature = self.pcnn(sample)
            out = self.outc(torch.cat([m_feature, i_feature_flip], dim=1))
            probs = F.softmax(out, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 2)
            result = torch.multinomial(probs, 1).float()
            sample = result.view(-1, 64, 64, 1).permute(0, 3, 1, 2)
            sample = sample * (1 - template) + initSample
            sample[:, :, 0:idx, :] = fixed

            sample = torch.flip(sample, [2])
            initSample = torch.flip(initSample, [2])
            template = torch.flip(template, [2])

        m_feature = self.pcnn(sample)
        i_logits = i_logits.repeat(n_sample, 1, 1, 1)
        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
        up_m_logits = self.up(m_logits)
    
        #logits = F.softmax(up_m_logits, dim=1)
        logits = F.softmax(self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), dim=1)

        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        logits = torch.multinomial(logits, 1).float()
        result = logits.view(-1, 128, 128, 1).permute(0, 3, 1, 2)
        result = kornia.morphology.closing(result, torch.ones(3, 3).to(device))
        return kornia.filters.median_blur(result, (3, 3))

    def interactive_samples_fast_(self, device, image, initSample, template, n_sample=8):
        def func(i, j, out, sample):
            probs = F.softmax(out, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 2)
            result = torch.multinomial(probs, 1).float()
            result = result.view(-1, 64, 64, 1).permute(0, 3, 1, 2)
            while i < 64:
                while j < 64:
                    if torch.sum(result[:, :, i, j]) == 0:
                        j = j + 1
                    else:
                        sample[:, :, i, j] = result[:, :, i, j]
                        return i, j, sample
                i = i + 1
                j = 0
            return i, j, sample
                    
        sample = initSample.repeat(n_sample, 1, 1, 1)
        i_logits, i_feature = self.unet(image)
        i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        i_logits = i_logits.repeat(n_sample, 1, 1, 1)
        image_ = torch.flip(image, [2])
        i_logits_, i_feature_flip = self.unet(image_)
        i_feature_flip = i_feature_flip.repeat(n_sample, 1, 1, 1)
        
        epoch = 2
        for n in range(epoch):
#            idx = torch.nonzero(torch.squeeze(initSample))[0, 0].item()
#            i = idx
#            j = torch.nonzero(torch.squeeze(initSample))[0, 1].item()
            i=0
            while i < 64:
#                if i != idx:
#                    j = 0
                j=0
                while j < 64:
                   if i == 64:
                       break
                   if torch.squeeze(template)[i, j] == 1.0:
                       sample[:, :, i, j] = initSample[:, :, i, j]
                   else:
                       m_feature = self.pcnn(sample)
                       out = self.outc(torch.cat([m_feature, i_feature], dim=1))
                       i, j, sample = func(i, j, out, sample)
                   j = j + 1
                i = i + 1
            
            sample = sample * (1 - template) + initSample
            
            sample = torch.flip(sample, [2])
            initSample = torch.flip(initSample, [2])
            template = torch.flip(template, [2])
#            
#            idx = torch.nonzero(torch.squeeze(initSample))[0, 0].item()
#            i = idx
#            j = torch.nonzero(torch.squeeze(initSample))[0, 1].item()
            i=0
            while i < 64:
#                if i != idx:
#                    j = 0
                j=0
                while j < 64:
                   if i == 64:
                       break
                   if torch.squeeze(template)[i, j] == 1.0:
                       sample[:, :, i, j] = initSample[:, :, i, j]
                   else:
                       m_feature = self.pcnn(sample)
                       out = self.outc(torch.cat([m_feature, i_feature], dim=1))
                       i, j, sample = func(i, j, out, sample)
                   j = j + 1
                i = i + 1
            
            sample = sample * (1 - template) + initSample

            sample = torch.flip(sample, [2])
            initSample = torch.flip(initSample, [2])
            template = torch.flip(template, [2])
            

        m_feature = self.pcnn(sample)
        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
        up_m_logits = self.up(m_logits)
        logits = F.softmax(self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), dim=1)
        return torch.argmax(logits, dim=1, keepdim=True)

    def interactive_samples_fast(self, device, image, initSample, template, mode, n_sample=8):
        def func(i, j, out, sample):
            probs = F.softmax(out, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 2)
            result = torch.multinomial(probs, 1).float()
            result = result.view(-1, 64, 64, 1).permute(0, 3, 1, 2)
            while i < 64:
                while j < 64:
                    if torch.sum(result[:, :, i, j]) == 0:
                        j = j + 1
                    else:
                        sample[:, :, i, j] = result[:, :, i, j]
                        return i, j, sample
                i = i + 1
                j = 0
            return i, j, sample
                    
        sample = initSample.repeat(n_sample, 1, 1, 1)
        i_logits, i_feature = self.unet(image)
        i_feature = i_feature.repeat(n_sample, 1, 1, 1)
        i_logits = i_logits.repeat(n_sample, 1, 1, 1)
        if mode == 1:
            fd=2
            image = torch.flip(image, [fd])
            i_logits, i_feature = self.unet(image)
            i_feature = i_feature.repeat(n_sample, 1, 1, 1)
            sample = torch.flip(sample, [fd])
            initSample = torch.flip(initSample, [fd])
            template = torch.flip(template, [fd])
        elif mode == 3:
            fd = 3
            image = torch.flip(image, [fd])
            i_logits, i_feature = self.unet(image)
            i_feature = i_feature.repeat(n_sample, 1, 1, 1)
            sample = torch.flip(sample, [fd])
            initSample = torch.flip(initSample, [fd])
            template = torch.flip(template, [fd])


        
        epoch = 1
        for n in range(epoch):
            i=0
            while i < 64:
                j=0
                while j < 64:
                   if i == 64:
                       break
                   if torch.squeeze(template)[i, j] == 1.0:
                       sample[:, :, i, j] = initSample[:, :, i, j]
                   else:
                       m_feature = self.pcnn(sample)
                       out = self.outc(torch.cat([m_feature, i_feature], dim=1))
                       i, j, sample = func(i, j, out, sample)
                   j = j + 1
                i = i + 1

        if mode == 1:
            fd=2
            image = torch.flip(image, [fd])
            i_logits, i_feature = self.unet(image)
            i_logits = i_logits.repeat(n_sample, 1, 1, 1)
            i_feature = i_feature.repeat(n_sample, 1, 1, 1)
            sample = torch.flip(sample, [fd])
            initSample = torch.flip(initSample, [fd])
            template = torch.flip(template, [fd])
        elif mode == 3:
            fd = 3
            image = torch.flip(image, [fd])
            i_logits, i_feature = self.unet(image)
            i_logits = i_logits.repeat(n_sample, 1, 1, 1)
            i_feature = i_feature.repeat(n_sample, 1, 1, 1)
            sample = torch.flip(sample, [fd])
            initSample = torch.flip(initSample, [fd])
            template = torch.flip(template, [fd])
            

        m_feature = self.pcnn(sample)
        m_logits = self.outc(torch.cat([m_feature, i_feature], dim=1))
        up_m_logits = self.up(m_logits)
        logits = F.softmax(self.final(torch.cat([i_logits, up_m_logits.detach()], dim=1)), dim=1)
        return torch.argmax(logits, dim=1, keepdim=True)
            
               
        
        
        
        
        
        
        