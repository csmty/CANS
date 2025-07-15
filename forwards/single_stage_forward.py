from utils.registry import FORWARD_REGISTRY
from fvcore.nn import FlopCountAnalysis, flop_count_table
from utils.crop_merge import crop_merge


@FORWARD_REGISTRY.register()
def ss_train_forward(config, model, data):
    raw = data['noisy_raw'].cuda(non_blocking=True)
    rgb_gt = data['clean_rgb'].cuda(non_blocking=True)
    
    rgb_out = model(raw)
    ###### | output          | label
    return {'rgb': rgb_out}, {'rgb': rgb_gt}

@FORWARD_REGISTRY.register()
def ss_raw_train_forward(config, model, data):
    raw = data['noisy_raw'].cuda(non_blocking=True)
    raw_gt = data['clean_raw'].cuda(non_blocking=True)
    
    raw_out = model(raw)
    ###### | output          | label
    return {'raw': raw_out}, {'raw': raw_gt}


@FORWARD_REGISTRY.register()
def ss_test_forward(config, model, data):
    if not config['test'].get('cpu', False):
        raw = data['noisy_raw'].cuda(non_blocking=True)
        rgb_gt = data['clean_rgb'].cuda(non_blocking=True)
    else:
        raw = data['noisy_raw']
        rgb_gt = data['clean_rgb']
    img_files = data['img_file']
    lbl_files = data['lbl_file']

    if 'crop' in config['test'] and config['test']['crop']:
        _,c,h,w=raw.shape
        cm=crop_merge(c,h,w)
        cropped=cm.eval_crop(raw)

        output_list=[]
        for c in cropped:
            output_list.append(c[None,...])
        cropped=torch.cat(output_list,dim=0)
        _,c,_,_=cropped.shape
        cm=crop_merge(c,h,w)
        rgb_out=cm.eval_merge(cropped)      
    else:
        rgb_out = model(raw)

    return {'rgb': rgb_out}, {'rgb': rgb_gt}, img_files, lbl_files

@FORWARD_REGISTRY.register()
def ss_raw_test_forward(config, model, data):
    if not config['test'].get('cpu', False):
        raw = data['noisy_raw'].cuda(non_blocking=True)
        raw_gt = data['clean_raw'].cuda(non_blocking=True)
    else:
        raw = data['noisy_raw']
        raw_gt = data['clean_raw']
    img_files = data['img_file']
    lbl_files = data['lbl_file']

    raw_out = model(raw)

    return {'raw': raw_out}, {'raw': raw_gt}, img_files, lbl_files


@FORWARD_REGISTRY.register()  # without label, for inference only
def ss_inference(config, model, data):
    raw = data['noisy_raw'].cuda(non_blocking=True)
    img_files = data['img_file']

    rgb_out = model(raw)

    ###### | output          | img names
    return {'rgb': rgb_out}, img_files




######################################################################################################

import torch
def crop_forward(x):
    B, C, H, W = x.shape
    padding = 20
    new_input = torch.zeros(size=(B, 4, C, H//2+padding, W//2+padding)).to(x.device)
    new_input[:,0,:,:,:] = x[..., 0:(H//2+padding), 0:(W//2+padding)]
    new_input[:,1,:,:,:] = x[..., 0:(H//2+padding), (W//2-padding):]
    new_input[:,2,:,:,:] = x[..., (H//2-padding):, 0:(W//2+padding)]
    new_input[:,3,:,:,:] = x[..., (H//2-padding):, (W//2-padding):]
    return new_input

def crop_backward(x, outputs, flag='raw'):
    B, C, H, W = x.shape
    if flag=='raw':
        output = torch.zeros(size=(B, C, H, W)).to(x.device)
        output[..., 0:H//2, 0:W//2] = outputs[0][..., 0:H//2, 0:W//2]
        output[..., 0:H//2, W//2:] = outputs[1][..., 0:H//2, 20:]
        output[..., H//2:, 0:W//2] = outputs[2][..., 20:, 0:W//2]
        output[..., H//2:, W//2:] = outputs[3][..., 20:, 20:]
    else:
        output = torch.zeros(size=(B, 3, 2*H, 2*W)).to(x.device)
        # print(outputs[0].shape)
        output[..., 0:H, 0:W] = outputs[0][..., 0:H, 0:W]
        output[..., 0:H, W:] = outputs[1][..., 0:H, 40:]
        output[..., H:, 0:W] = outputs[2][..., 40:, 0:W]
        output[..., H:, W:] = outputs[3][..., 40:, 40:]
    return output


@FORWARD_REGISTRY.register()
def ss_test_forward_patch(config, model, data):
    if not config['test'].get('cpu', False):
        raw = data['noisy_raw'].cuda(non_blocking=True)
        rgb_gt = data['clean_rgb'].cuda(non_blocking=True)
    else:
        raw = data['noisy_raw']
        rgb_gt = data['clean_rgb']
    img_files = data['img_file']
    lbl_files = data['lbl_file']

    new_input = crop_forward(raw)
    outputs_rgb = []
    for i in range(new_input.shape[2]):
        rgb_out = model(new_input[:,i,...])
        outputs_rgb.append(rgb_out)
    rgb_out = crop_backward(raw, outputs_rgb, 'rgb')
    return {'rgb': rgb_out}, {'rgb': rgb_gt}, img_files, lbl_files


######################################################################################################
