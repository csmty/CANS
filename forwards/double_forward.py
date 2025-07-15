from utils.registry import FORWARD_REGISTRY
from fvcore.nn import FlopCountAnalysis, flop_count_table


@FORWARD_REGISTRY.register(suffix='DNF')
def dd_train_forward(config, model, data):
    raw = data['noisy_raw'].cuda(non_blocking=True)
    rgb_gt = data['clean_rgb'].cuda(non_blocking=True)
    rgb_gt_sharp = data['clean_rgb_sharp'].cuda(non_blocking=True)
    freq_gt = data['freqency'].cuda(non_blocking=True)
    
    rgb_out, rgb_out_main, rgb_out_freq = model(raw)
    ###### | output                          | label
    return {'rgb': rgb_out, 'main': rgb_out_main, 'freq':rgb_out_freq}, {'rgb': rgb_gt_sharp, 'main': rgb_gt, 'freq':freq_gt}


@FORWARD_REGISTRY.register(suffix='DNF')
def dd_test_forward(config, model, data):
    if not config['test'].get('cpu', False):
        raw = data['noisy_raw'].cuda(non_blocking=True)
        rgb_gt = data['clean_rgb'].cuda(non_blocking=True)
        rgb_gt_sharp = data['clean_rgb_sharp'].cuda(non_blocking=True)
        freq_gt = data['freqency'].cuda(non_blocking=True)
    else:
        raw = data['noisy_raw']
        rgb_gt = data['clean_rgb']
        rgb_gt_sharp = data['clean_rgb_sharp']
        freq_gt = data['freqency']

    img_files = data['img_file']
    lbl_files = data['lbl_file']

    rgb_out, rgb_out_main, rgb_out_freq = model(raw)

    ###### | output                          | label                         | img and label names
    return {'rgb': rgb_out, 'main': rgb_out_main, 'freq':rgb_out_freq}, {'rgb': rgb_gt_sharp, 'main': rgb_gt, 'freq':freq_gt}, img_files, lbl_files


@FORWARD_REGISTRY.register(suffix='DNF')  # without label, for inference only
def dd_inference(config, model, data):
    raw = data['noisy_raw'].cuda(non_blocking=True)
    img_files = data['img_file']

    rgb_out, rgb_out_main, rgb_out_freq = model(raw)

    ###### | output                          | img names
    return {'rgb': rgb_out, 'main': rgb_out_main, 'freq':rgb_out_freq}, img_files
