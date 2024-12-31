# MSAdaptor

## mindtorch

### How to use

#### install from source

```bash
git clone https://github.com/lvyufeng/msadaptor
cd msadaptor
bash scripts/build_and_reinstall.sh
# remove torch, torch_npu, torchvision automaticlly
# and install mindtorch.
```

#### install from pip

```bash
pip uninstall torch torch_npu torchvision -y
pip install msadaptor
```