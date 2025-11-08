
import torch
from collections import OrderedDict
from config.config import parse_args_yml



def remove_filed_from_state_dict(state_dict):
    """
    遍历模型的 state_dict，将所有包含 'module' 的部分去掉。

    参数:
        state_dict (OrderedDict): 模型的 state_dict。

    返回:
        OrderedDict: 修改后的 state_dict。
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("feature_extractor.", "")
        new_state_dict[new_key] = value
    return new_state_dict

def processing_teacher_backbone():
    
    abs_path = 'logs/teacher_resnet20_cifar100/version_0'
    # abs_path = 'logs/teacher_swin_cars/version1'
    args = parse_args_yml(f'{abs_path}/hparams.yaml')
    pre_trained_path = f'{abs_path}/checkpoints/best_model.ckpt'
    save_new_checkpoint_path = f'pre_trained_ann_model/{args.arch}_{args.data_type}.pth'
    state_dict = torch.load(pre_trained_path, map_location='cpu')
    new_state = remove_filed_from_state_dict(state_dict['state_dict'])
    new_checkpoint = {
        'state_dict':new_state
    }
    # Save the new checkpoint
    torch.save(new_checkpoint, save_new_checkpoint_path)

# process_teacher_backbone()

processing_teacher_backbone()
# processing_teacher_multistage()
# /python3 plt_train_kd_ckd_snn.py --config config/cifar10/plt_train_kd_ckd_snn20.yml