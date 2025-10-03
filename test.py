import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
import random
from collections import OrderedDict
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
  'ForenSynths': {
    'dataroot': './dataset/ForenSynths/',
    'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
    'no_crop': True,
  },
  'GANGen-Detection': {
    'dataroot': './dataset/GANGen-Detection/',
    'no_resize': True,
    'no_crop': True,
  },
  'DiffusionForensics': {
    'dataroot': './dataset/DiffusionForensics/',
    'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
    'no_crop': True,
  },
  'UniversalFakeDetect': {
    'dataroot': './dataset/UniversalFakeDetect/',
    'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
    'no_crop': True,
  },
  'Chameleon': {
    'dataroot': './dataset/Chameleon/',
    'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
    'no_crop': True,
  },
}


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')
opt.model_path = "/root/NPR-DeepfakeDetection/NPR.pth"  # 绝对路径测试
print(opt.model_path)
# get model
model = resnet50(num_classes=1)
#model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
#checkpoint = torch.load(opt.model_path, map_location='cpu')
#model.load_state_dict(checkpoint['model'], strict=True)
checkpoint = torch.load(opt.model_path, map_location='cpu')
state_dict = checkpoint['model']  # 或 checkpoint 本身，如果是整个 dict
# 去掉 DataParallel 的 "module." 前缀
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # 去掉 module.
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=True)
model.cuda()
model.eval()

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = [];aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, _, _, _, _ = validate(model, opt)
        accs.append(acc);aps.append(ap)
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

