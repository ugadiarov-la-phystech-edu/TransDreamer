import torch

from ocr.dinov2saur.dinov2.models import build_model_from_cfg


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # remove `dino_head.` parameters
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dino_head.")}
    msg = model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def build_model_for_eval(config, checkpoint_path, need_save_qkv_output_last_layers=0):
    config['student']['need_save_qkv_output_last_layers'] = need_save_qkv_output_last_layers
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if checkpoint_path is not None:
        load_pretrained_weights(model, checkpoint_path, checkpoint_key='teacher')

    model.eval()
    model.requires_grad_(False)
    model.cuda()
    return model
