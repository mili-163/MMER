# Model Directory

这个目录用于存放多模态情感识别模型的实现。

## 已删除的文件
- `imder.py` - IMDer模型实现（已删除）
- `scoremodel.py` - 扩散模型实现（已删除）  
- `rcan.py` - 残差通道注意力网络（已删除）

## 需要实现
请在此目录下实现您的新模型，并确保：
1. 模型类继承自 `torch.nn.Module`
2. 实现 `forward` 方法
3. 在 `trains/singleTask/__init__.py` 中导入您的模型
4. 在 `trains/ATIO.py` 中注册您的训练器 