import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  TRAINING ON CPU - This is VERY slow!')
    print('   Install PyTorch with CUDA support for 10-50x speedup')
