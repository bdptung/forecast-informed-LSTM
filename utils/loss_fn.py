import numpy as np
import torch
from torch import hypot
from torch.nn import Module as nnModule
from torch.nn.functional import smooth_l1_loss

class LossFN(nnModule):
  def __init__(self, losstype, scalers, eps=1e-8, grad_weight=0.2, delta=1.0):
    super().__init__()
    self.losstype = losstype.lower()
    self.scalers = scalers
    self.eps = eps
    self.grad_weight = grad_weight
    self.delta = delta   # Huber transition point; scalar, or per-target sequence (len K)

  def forward(self, pred, true, mask, with_grad_penalty=True):
    # with_grad_penalty: the gradient/shape term is a TRAINING-only auxiliary. Eval passes
    # False so the validation loss is pure prediction error - comparable across runs and the
    # right quantity to select/early-stop on.
    loss = 0
    if self.losstype == 'mae':
        loss += self.masked_mae_multi(pred, true, mask)
    elif self.losstype == 'mse':
        loss += self.masked_mse_multi(pred, true, mask)
    elif self.losstype == 'huber':
        loss += self.masked_huber_multi(pred, true, mask)
    if with_grad_penalty and self.grad_weight > 0:
      loss += self.gradient_penalty_multi(pred, true, mask, loss_type=self.losstype)
    return loss

  def gradient_penalty(self, pred, true, mask, loss_type='mse'):
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    true_diff = true[:, 1:, :] - true[:, :-1, :]
    mask_diff = mask[:, 1:, :] * mask[:, :-1, :]
    diff = (pred_diff - true_diff) * mask_diff
    if loss_type == 'mae':
        diff = diff.abs()
    elif loss_type == 'mse':
        diff = diff ** 2
    grad_loss = diff.sum() / mask_diff.sum().clamp_min(self.eps)
    return self.grad_weight * grad_loss
  
  def gradient_penalty_multi(self, pred, true, mask, loss_type='mse'):
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    true_diff = true[:, 1:, :] - true[:, :-1, :]
    mask_diff = mask[:, 1:, :] * mask[:, :-1, :]
    diff = (pred_diff - true_diff) * mask_diff
    if loss_type == 'mae':
        diff = diff.abs()
    elif loss_type == 'mse':
        diff = diff ** 2
    elif loss_type == 'huber':
        diff = diff.abs()
    grad_loss = diff.sum(dim=(0,1)) / mask_diff.sum(dim=(0,1)).clamp_min(self.eps)
    return self.grad_weight * grad_loss.mean()

  def masked_mae(self, pred, true, mask, eps=1e-8):
      """
      pred, true, mask: same shape (B,H) or (B,H,K)
      mask: 1=label present, 0=missing
      """
      diff = (pred - true)
      num = (diff.abs() * mask).sum()
      den = mask.sum().clamp_min(eps)
      return num / den

  def masked_mae_multi(self, pred, true, mask, eps=1e-8):
      """
      pred, true, mask: same shape (B,H) or (B,H,K)
      mask: 1=label present, 0=missing
      """
      diff = (pred - true)
      ae = (diff.abs() * mask).sum(dim=(0,1))
      den = mask.sum(dim=(0,1)).clamp_min(eps)
      mae_per_target = ae / den
      mean_mae = mae_per_target.mean()
      return mean_mae

  def masked_mse(self, pred, true, mask, eps=1e-8):
      """
      pred, true, mask: same shape (B,H) or (B,H,K)
      mask: 1=label present, 0=missing
      """
      diff = (pred - true) ** 2
      num = (diff * mask).sum()
      den = mask.sum().clamp_min(eps)
      return num / den

  def masked_mse_multi(self, pred, true, mask, eps=1e-8):
      """
      pred, true, mask: same shape (B,H) or (B,H,K)
      mask: 1=label present, 0=missing
      """
      diff = (pred - true) ** 2
      se = (diff * mask).sum(dim=(0,1))
      den = mask.sum(dim=(0,1)).clamp_min(eps)
      mse_per_target = se / den
      mean_mse = mse_per_target.mean()
      # print(f'      mse_ [{', '.join(format(x, ".5f") for x in mse_per_target.tolist())}] {mean_mse.item():2f}')
      return mean_mse
  
  def masked_huber_multi(self, pred, true, mask, eps=1e-8):
      """
      pred, true, mask: same shape (B,H) or (B,H,K)
      mask: 1=label present, 0=missing
      Smooth-L1 form of Huber with a configurable transition point self.delta (scalar, or a
      per-target sequence broadcast over the last dim). Unlike torch's huber (which scales the
      whole loss by delta -> tiny values and ±delta linear-region gradients), smooth-L1 divides
      the quadratic branch by delta so the loss VALUE and the linear-region GRADIENT (±1) stay
      on the MAE scale, keeping numbers comparable to MAE runs and the tuned LR valid:
        quadratic 0.5*err^2/delta for |err| <= delta, linear |err| - 0.5*delta beyond.
      """
      err = pred - true
      ae = err.abs()
      delta = torch.as_tensor(self.delta, device=pred.device, dtype=pred.dtype).clamp_min(eps)  # scalar or (K,)
      sl1_errors = torch.where(ae <= delta, 0.5 * err.pow(2) / delta, ae - 0.5 * delta)
      sl1 = (sl1_errors * mask).sum(dim=(0,1))
      den = mask.sum(dim=(0,1)).clamp_min(eps)
      return (sl1 / den).mean()
  

class LossFN_winddir(nnModule):
  """Loss for wind direction expressed as exactly two channels: [sin_dir, cos_dir].
  Both pred and true are (B, H, 2). Rather than penalising sin/cos independently,
  it measures the angular error between the predicted and true direction vectors:
  both are normalised to unit vectors and the signed angle between them is taken via
  atan2, so wraparound (359 deg vs 1 deg) costs ~0 instead of a large element-wise error.

  losstype:
    'mae'    -> mean |Δθ|         (radians; x180/pi for degrees)
    'mse'    -> mean Δθ^2
    'huber'  -> huber on Δθ
    'cosine' -> mean (1 - cos Δθ)  (smooth, atan2-free)
  """
  def __init__(self, losstype, scalers, eps=1e-8, grad_weight=0.0, delta=1.0):
    super().__init__()
    self.losstype = losstype.lower()
    self.scalers = scalers
    self.eps = eps
    self.grad_weight = grad_weight
    self.delta = delta   # Huber transition point on the angular error Δθ (radians)

  def _unit(self, v):
    norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
    return v / norm

  def forward(self, pred, true, mask, with_grad_penalty=True):
    # with_grad_penalty accepted for interface parity with LossFN (this loss has no grad term).
    # pred, true: (B, H, 2) = [sin_dir, cos_dir]; mask: (B, H, 2) or (B, H)
    p = self._unit(pred)
    t = self._unit(true)
    sin_p, cos_p = p[..., 0], p[..., 1]
    sin_t, cos_t = t[..., 0], t[..., 1]
    cos_delta = (sin_p * sin_t + cos_p * cos_t).clamp(-1.0, 1.0)  # (B, H)
    sin_delta = sin_p * cos_t - cos_p * sin_t                     # (B, H)
    delta = torch.atan2(sin_delta, cos_delta)                     # signed Δθ in [-pi, pi]

    # collapse the per-channel mask to one weight per timestep (sin/cos share availability)
    m = mask.amin(dim=-1) if mask.dim() == pred.dim() else mask   # (B, H)
    den = m.sum().clamp_min(self.eps)

    if self.losstype == 'mae':
        err = delta.abs()
    elif self.losstype == 'mse':
        err = delta ** 2
    elif self.losstype == 'huber':
        # smooth-L1 on the angular error Δθ (beta=self.delta is the transition), so the value
        # stays on the radian/MAE scale. `delta` here is the Δθ tensor, not the threshold.
        err = smooth_l1_loss(delta, torch.zeros_like(delta), reduction='none', beta=self.delta)
    elif self.losstype == 'cosine':
        err = 1.0 - cos_delta
    else:
        err = delta.abs()
    return (err * m).sum() / den
  
def masked_mae_multi(pred, true, mask, eps=1e-8):
    """
    pred, true, mask: same shape (B,H) or (B,H,K)
    mask: 1=label present, 0=missing
    """
    diff = (pred - true)
    ae = (diff.abs() * mask).sum(dim=(0,1))
    den = mask.sum(dim=(0,1)).clamp_min(eps)
    mae_per_target = ae / den
    mean_mae = mae_per_target.mean()
    return mean_mae
