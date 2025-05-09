REM filepath: d:\BTLT\Security\UnknownDetectionSystemFinal\install.bat
@echo off
echo === Cai dat dependencies vao moi truong .conda/ ===

:: Kich hoat moi truong conda
call conda activate .conda/
if %ERRORLEVEL% NEQ 0 (
    echo Khong the kich hoat moi truong conda .conda/
    exit /b 1
)
echo Moi truong conda .conda/ da duoc kich hoat.
echo.

echo === 1. Cai dat numpy va tensorflow ===
pip install numpy==1.26.4
pip install tensorflow==2.10.*
pip install keras
echo.

echo === 2. Cai dat CUDA va cuDNN ===
call conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
echo.

echo === 3. Cai dat deepface ===
pip install deepface
echo.

echo === 4. Cai dat dlib ===
call conda install -y -c conda-forge dlib
echo.

echo === 5. Cai dat cac goi phu thuoc ===
pip install certifi click dotmap face-recognition face-recognition-models imutils requests
pip install pillow==10.2.0
pip install protobuf==3.19.6
pip install tqdm opencv-python scikit-image
echo.

echo === 6. Cai dat CUDA toolkit va thu vien bo sung ===
call conda install -y -c conda-forge cudatoolkit-dev
call conda install -y -c conda-forge hnswlib
call conda install -y -c conda-forge scikit-learn
echo.

echo === 7. Cai dat PyTorch voi CUDA ===
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
echo.

echo === 8. Cai dat ONNX Runtime GPU ===
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
pip install onnx==1.12.0 bytetracker-gml
call conda install -y -c conda-forge nomkl
echo.

echo === 9. Cai dat cac goi FastAPI va WebRTC ===
pip install fastapi uvicorn[standard] jinja2 websockets python-multipart
pip install aiortc==1.11.0 aiohttp==3.11.18 av==14.3.0
echo.

echo === Cai dat hoan thanh trong moi truong .conda/ ===
pause