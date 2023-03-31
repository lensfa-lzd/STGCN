import os
import torch


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# g = os.walk('.')
# for path, dir_list, file_list in g:
#     for file_name in file_list:
#         if 'pth' in file_name:
#             checkpoint = torch.load(file_name, map_location='cpu')
#             loss = checkpoint['loss(mae)']
#             print(file_name, ' ', loss)

checkpoint = torch.load('Kt_2_Ks_2_15min_ckpt.pth', map_location='cpu')
config_args = checkpoint['config_args']
loss = checkpoint['loss(mae)']
# print('loss', '=', loss)

config_args.gso = None
# print(config_args)

for key, value in vars(config_args).items():
    if 'mask' not in key:
        print(key, '=', value)


# checkpoint_ = torch.load('head_4_channel_16_15min_pems_cat.pth')
# loss = checkpoint_['val_loss(mae)']
#
# checkpoint = {
#         'config_args': checkpoint_['config_args'],
#         'net': checkpoint_['net'],
#         'loss(mae)': loss,
#     }
#
# torch.save(checkpoint, 'head_4_channel_16_15min_pems_cat_.pth')