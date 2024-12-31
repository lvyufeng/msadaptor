pip uninstall torch torch_npu torchvision -y

rm -rf ./dist
python setup.py bdist_wheel
rm -rf *.egg-info
pip uninstall msadaptor -y && pip install dist/*.whl