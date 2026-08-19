set -e
CONDA=/home/glaze/miniconda3/bin/conda
PY=/home/glaze/miniconda3/envs/phywam/bin/python
echo "### 1/2 conda: gz python bindings"
$CONDA install -y -n phywam -c conda-forge gz-transport13-python gz-msgs10-python
echo "### 2/2 pip: torch + gui + vision"
$PY -m pip install --upgrade pip
$PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install opencv-python-headless PySide6 pyqtgraph imageio imageio-ffmpeg \
    matplotlib tensorboard einops scikit-image psutil pynvml
echo "### DONE"
