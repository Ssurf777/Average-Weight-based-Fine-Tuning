# awft_util.py
import torch

def load_scripted_model(path, device='cpu'):
    """
    TorchScript形式のモデルを読み込みます。
    """
    return torch.jit.load(path, map_location=device)

def apply_awft_hooks(model_from_scripted, new_model):
    """
    AWFT (Average Weight-based Fine-Tuning) を適用し、
    新しいモデルにパラメータを転送しつつ、重みの一部を凍結します。

    Parameters:
        model_from_scripted: 事前学習済みモデル (TorchScript形式)
        new_model: 新しいtorch.nn.Moduleインスタンス

    Returns:
        new_model: AWFT hooksが適用された新しいモデル
        mask_log: パラメータごとの訓練/凍結数ログ（dict）
    """
    masks = {}
    mask_log = {}

    # マスク作成
    for name, param in model_from_scripted.named_parameters():
        if 'weight' in name:
            mean_abs_weight = torch.mean(torch.abs(param))
            mask = (torch.abs(param) < mean_abs_weight)
            masks[name] = mask
        elif 'bias' in name:
            param.requires_grad = True

    # 重みのコピーとhookの登録
    for (new_name, new_param), (old_name, old_param) in zip(new_model.named_parameters(), model_from_scripted.named_parameters()):
        new_param.data = old_param.data.clone()

        if 'weight' in new_name and new_name in masks:
            mask = masks[new_name]
            def hook_factory(mask):
                return lambda grad: grad * mask.float()
            new_param.register_hook(hook_factory(mask))
            
            # ログ保存
            mask_log[new_name] = {
                'trainable': int(mask.sum().item()),
                'frozen': int((~mask).sum().item()),
                'mean_abs': float(torch.mean(torch.abs(new_param)).item())
            }

    return new_model, mask_log
