# Libraries.
import numpy as np
import torch
import torch.nn as nn

# Stem module of the CMT architecture.
class Stem(nn.Module):
    
  # Constructor.
  def __init__(self, in_channels, out_channels):

    super().__init__()

    # Conv 3 x 3, stride 2.
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
    self.bn1 = nn.BatchNorm2d(out_channels)

    # Conv 3 x 3. 
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # Conv 3 x 3.
    self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
    self.bn3 = nn.BatchNorm2d(out_channels)

    # GELU activation.
    self.gelu = nn.GELU()

  # Forward pass.
  def forward(self, x):
    x = self.conv1(x)
    x = self.gelu(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.gelu(x)
    x = self.bn2(x)
    x = self.conv3(x)
    x = self.gelu(x)
    y = self.bn3(x)
    return y
	
# Reduced stem module of the CMT architecture.
class ReducedStem(nn.Module):
    
  # Constructor.
  def __init__(self, in_channels, out_channels):

    super().__init__()

    # Conv 3 x 3, stride 2.
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
    self.bn = nn.BatchNorm2d(out_channels)

    # GELU activation.
    self.gelu = nn.GELU()

  # Forward pass.
  def forward(self, x):
    x = self.conv(x)
    x = self.gelu(x)
    y = self.bn(x)
    return y
	
# Patch embedding module of the CMT architecture.
class PatchEmbedding(nn.Module):

  # Constructor.
  def __init__(self, in_channels, out_channels):

    super().__init__()

    # Conv 2 x 2, stride 2.
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0)

  # Forward pass.
  def forward(self, x):
    x = self.conv(x)
    _, c, h, w = x.size()
    y = torch.nn.functional.layer_norm(x, [c, h, w])
    return y
	
# Local Perception Unit: LPU(X) = DWConv(X) + X.
class LPU(nn.Module):

  # Constructor.
  def __init__(self, in_channels, out_channels):

    super().__init__()

    # Depthwise convolution.
    self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, groups = in_channels)

  # Forward pass.
  def forward(self, x):
    y = self.dwconv(x) + x
    return y
	
# Standard Multi-Head-Self-Attention.
class MHSA(nn.Module):
    
  # Constructor.
  def __init__(self, d, d_k, d_v, heads):

    super().__init__()

    # Projection matrices.
    self.fc_q = nn.Linear(d, heads * d_k)
    self.fc_k = nn.Linear(d, heads * d_k)
    self.fc_v = nn.Linear(d, heads * d_v)
    self.fc_o = nn.Linear(heads * d_k, d)

    self.d = d
    self.d_k = d_k
    self.d_v = d_v
    self.heads = heads
    
  # Forward pass.
  def forward(self, x):

    # Extracting shape from input signal.
    b, c, h, w = x.shape

    # Layer normalization.
    x_norm = nn.functional.layer_norm(x, [c, h, w])

    # Reshaping and permuting x. Final shape is (b, h * w, c).
    x_reshape = x_norm.view(b, c, h * w).permute(0, 2, 1)

    # Getting queries by applying the fc_q linear projection.
    q = self.fc_q(x_reshape)

    # Reshaping and permuting the queries. Final shape is (b, heads, n = h * w, d_k). 
    q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

    # Projecting through fc_k.
    k = self.fc_k(x_reshape)

    # Reshaping and permuting the keys. Final shape is (b, heads, h * w, d_k).
    k = k.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

    # Projecting through fc_v.
    v = self.fc_v(x_reshape)

    # Reshaping and permuting the keys. Final shape is (b, heads, h * w, d_v).
    v = v.view(b, h * w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous()

    # Computing softmax((Q K^T) / sqrt(d_k)).
    attention = torch.einsum("... i d, ... j d -> ... i j", q, k) * (self.d_k ** -0.5)
    attention = torch.softmax(attention, dim = -1)

    # Applying attention scores to values by taking the dot product.
    tmp = torch.matmul(attention, v).permute(0, 2, 1, 3)

    # Permuting the result. Final shape is (b, n = h * w, heads, d_v).
    tmp = tmp.contiguous().view(b, h * w, self.heads * self.d_v)

    # Projecting using fc_o and reshaping. Final shape is (b, c, h, w).
    tmp = self.fc_o(tmp).view(b, self.d, h, w)

    # Returning tmp + x (skip connection).
    return tmp + x
	
# Lightweight Multi-Head-Self-Attention.
class LMHSA(nn.Module):
    
  # Constructor.
  def __init__(self, input_size, d, d_k, d_v, stride, heads):

    super().__init__()

    # Depthwise convolutions.
    self.dwconv = nn.Conv2d(d, d, kernel_size = stride, stride = stride, groups = d)

    # Projection matrices.
    self.fc_q = nn.Linear(d, heads * d_k)
    self.fc_k = nn.Linear(d, heads * d_k)
    self.fc_v = nn.Linear(d, heads * d_v)
    self.fc_o = nn.Linear(heads * d_k, d)

    self.d = d
    self.d_k = d_k
    self.d_v = d_v
    self.heads = heads

    # Relative position bias to each self-attention module. Shape is n x n/k^2, where n = h * w.
    self.B = nn.Parameter(torch.randn(1, self.heads, input_size ** 2, (input_size // stride) ** 2), requires_grad = True)
    
  # Forward pass.
  def forward(self, x):

    # Extracting shape from input signal.
    b, c, h, w = x.shape

    # Layer normalization.
    x_norm = nn.functional.layer_norm(x, [c, h, w])

    # Reshaping and permuting x. Final shape is (b, h * w, c).
    x_reshape = x_norm.view(b, c, h * w).permute(0, 2, 1)

    # Getting queries by applying the fc_q linear projection.
    q = self.fc_q(x_reshape)

    # Reshaping and permuting the queries. Final shape is (b, heads, n = h * w, d_k). 
    q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

    # Applying depthwise conv to x.
    k = self.dwconv(x)

    # Extracting shape of keys.
    k_b, k_c, k_h, k_w = k.shape

    # Reshaping and permuting keys. Final shape is (k_b, k_h * k_w, k_c).
    k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()

    # Projecting through fc_k.
    k = self.fc_k(k)

    # Reshaping and permuting the keys. Final shape is (k_b, heads, k_h * k_w, d_k).
    k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

    # Applying depthwise conv to x.
    v = self.dwconv(x)

    # Extracting shape of values.
    v_b, v_c, v_h, v_w = v.shape

    # Reshaping and permuting values. Final shape is (v_b, v_h * v_w, v_c).
    v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()

    # Projecting through fc_v.
    v = self.fc_v(v)

    # Reshaping and permuting the keys. Final shape is (v_b, heads, v_h * v_w, d_v).
    v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous()

    # Computing softmax((Q K'^T) / sqrt(d_k) + B).
    attention = torch.einsum("... i d, ... j d -> ... i j", q, k) * (self.d_k ** -0.5)
    attention = attention + self.B
    attention = torch.softmax(attention, dim = -1)

    # Applying attention scores to values by taking the dot product.
    tmp = torch.matmul(attention, v).permute(0, 2, 1, 3)

    # Permuting the result. Final shape is (b, n = h * w, heads, d_v).
    tmp = tmp.contiguous().view(b, h * w, self.heads * self.d_v)

    # Projecting using fc_o and reshaping. Final shape is (b, c, h, w).
    tmp = self.fc_o(tmp).view(b, self.d, h, w)

    # Returning tmp + x (skip connection).
    return tmp + x
	
# Inverted Residual Feed-forward Network: IRFFN(X) = Conv(F(Conv(X))), F(X) = DWConv(X) + X.
class IRFFN(nn.Module):

  # Constructor.
  def __init__(self, in_channels, R):

    super().__init__()

    # Number of channels after expansion.
    out_channels = int(in_channels * R)

    # Conv 1 x 1.
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
    self.bn1 = nn.BatchNorm2d(out_channels)

    # Depthwise convolution.
    self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, groups = out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # Conv 1 x 1.
    self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size = 1)
    self.bn3 = nn.BatchNorm2d(in_channels)

    # GELU activation.
    self.gelu = nn.GELU()

  # Forward pass.
  def forward(self, x):
    b, c, h, w = x.shape
    tmp = nn.functional.layer_norm(x, [c, h, w])
    tmp = self.conv1(tmp)
    tmp = self.bn1(tmp)
    tmp = self.gelu(tmp)
    tmp = self.dwconv(tmp)
    tmp = self.bn2(tmp)
    tmp = self.gelu(tmp)
    tmp = self.conv2(tmp)
    tmp = self.bn3(tmp)
    y = x + tmp
    return y
	
# CMT block of the CMT architecture.
class CMTBlock(nn.Module):

  # Constructor.
  def __init__(self, input_size, k, d_k, d_v, heads, R, channels, attention_type):

    super().__init__()

    # Local Perception Unit.
    self.lpu = LPU(channels, channels)

    # Lightweight MHSA.
    if attention_type == "light":
      self.mhsa = LMHSA(input_size, channels, d_k, d_v, k, heads)
    
    # Standard MHSA.
    elif attention_type == "standard":
      self.mhsa = MHSA(channels, d_k, d_v, heads)
    
    # No attention.
    else:
      self.mhsa = None

    # Inverted Residual FFN.
    self.irffn = IRFFN(channels, R)

  # Forward pass.
  def forward(self, x):
    x = self.lpu(x)
    if self.mhsa != None: 
      x = self.mhsa(x)
    y = self.irffn(x)
    return y
	
# CMT architecture.
class CMT(nn.Module):

  # Constructor.
  def __init__(self, in_channels, stem_channels, channels, block_layers, k, heads, R, input_size, classes, attention_type):

    super(CMT, self).__init__()

    # Stem layer
    self.stem = Stem(in_channels, stem_channels)

    # Patch embedding.
    self.pe1 = PatchEmbedding(stem_channels, channels[0])
    self.pe2 = PatchEmbedding(channels[0], channels[1])
    self.pe3 = PatchEmbedding(channels[1], channels[2])
    self.pe4 = PatchEmbedding(channels[2], channels[3])

    # Stages.
    stage1 = [CMTBlock(input_size // 4,  k[0], channels[0] // heads[0], channels[0] // heads[0], heads[0], R, channels[0], attention_type) for _ in range(block_layers[0])]
    stage2 = [CMTBlock(input_size // 8,  k[1], channels[1] // heads[1], channels[1] // heads[1], heads[1], R, channels[1], attention_type) for _ in range(block_layers[1])]
    stage3 = [CMTBlock(input_size // 16, k[2], channels[2] // heads[2], channels[2] // heads[2], heads[2], R, channels[2], attention_type) for _ in range(block_layers[2])]
    stage4 = [CMTBlock(input_size // 32, k[3], channels[3] // heads[3], channels[3] // heads[3], heads[3], R, channels[3], attention_type) for _ in range(block_layers[3])]

    self.stage1 = nn.Sequential(*stage1)
    self.stage2 = nn.Sequential(*stage2)
    self.stage3 = nn.Sequential(*stage3)
    self.stage4 = nn.Sequential(*stage4)

    # Global average pooling.
    self.avg_pool = nn.AdaptiveAvgPool2d(1)

    # Projection layer.
    self.projection = nn.Sequential(
      nn.Conv2d(channels[3], 1280, kernel_size = 1),
      nn.ReLU(inplace = True),
    )

    # Classifier.
    self.classifier = nn.Linear(1280, classes)

  # Forward pass.
  def forward(self, x):
    x = self.stem(x)
    x = self.pe1(x)
    x = self.stage1(x)
    x = self.pe2(x)
    x = self.stage2(x)
    x = self.pe3(x)
    x = self.stage3(x)
    x = self.pe4(x)
    x = self.stage4(x)
    x = self.avg_pool(x)
    x = self.projection(x)
    x = torch.flatten(x, 1)
    y = self.classifier(x)
    return y
	
# Reduced CMT architecture.
class ReducedCMT(nn.Module):

  # Constructor.
  def __init__(self, in_channels, stem_channels, channels, block_layers, k, heads, R, input_size, classes, attention_type):

    super(ReducedCMT, self).__init__()

    # Stem layer
    self.stem = ReducedStem(in_channels, stem_channels)

    # Patch embedding.
    self.pe1 = PatchEmbedding(stem_channels, channels[0])
    self.pe2 = PatchEmbedding(channels[0], channels[1])
    self.pe3 = PatchEmbedding(channels[1], channels[2])

    # Stages.
    stage1 = [CMTBlock(input_size // 2, k[0], channels[0] // heads[0], channels[0] // heads[0], heads[0], R, channels[0], attention_type) for _ in range(block_layers[0])]
    stage2 = [CMTBlock(input_size // 4, k[1], channels[1] // heads[1], channels[1] // heads[1], heads[1], R, channels[1], attention_type) for _ in range(block_layers[1])]
    stage3 = [CMTBlock(input_size // 8, k[2], channels[2] // heads[2], channels[2] // heads[2], heads[2], R, channels[2], attention_type) for _ in range(block_layers[2])]

    self.stage1 = nn.Sequential(*stage1)
    self.stage2 = nn.Sequential(*stage2)
    self.stage3 = nn.Sequential(*stage3)

    # Global average pooling.
    self.avg_pool = nn.AdaptiveAvgPool2d(1)

    # Projection layer.
    self.projection = nn.Sequential(
      nn.Conv2d(channels[2], 256, kernel_size = 1),
      nn.ReLU(inplace = True),
    )

    # Classifier.
    self.classifier = nn.Linear(256, classes)

  # Forward pass.
  def forward(self, x):
    x = self.stem(x)
    x = self.pe1(x)
    x = self.stage1(x)
    x = self.pe2(x)
    x = self.stage2(x)
    x = self.pe3(x)
    x = self.stage3(x)
    x = self.avg_pool(x)
    x = self.projection(x)
    x = torch.flatten(x, 1)
    y = self.classifier(x)
    return y