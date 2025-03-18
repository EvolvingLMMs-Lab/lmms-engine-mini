
git clone https://github.com/NVIDIA/apex
cd apex
python3 -m pip install packagine
# pip install -v --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
python3 setup.py install
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.5.0
python3 -m pip install pybind11
python3 -m pip install --no-use-pep517 -e .

python3 -m pip install -v git+https://github.com/NVIDIA/TransformerEngine.git@stable
