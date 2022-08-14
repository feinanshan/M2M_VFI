# Video Frame Interpolation with M2M

### 1. Prepare Testing Data

- **Vimeo90K**: Download [Vimeo90K dataset](http://toflow.csail.mit.edu/) 

- **UCF101**: Download [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow) 

- **ATD12K**: Download [ATD12K dataset](https://drive.google.com/file/d/1XBDuiEgdd6c0S4OXLF4QvgSn_XNPwc-g/view) 

- **Xiph**: Download [Xiph dataset](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark.py) 

- **X4K1000FPS**: Download [X4K1000FPS dataset](https://www.dropbox.com/sh/duisote638etlv2/AABJw5Vygk94AWjGM4Se0Goza?dl=0) 


### 2. Preparation:
- Download the [pretrained model](https://drive.google.com/file/d/1dO-ArTLJ4cMZuN6dttIFFMLtp4I2LnSG/view?usp=sharing) or copy the trained model to `./Test`

- Modify the path to dataset `strPath` in `bench_*.py` accordingly.


- Install enviornment
```
pip install -r requirements.txt
```

### 3. Evaluation:

Evaluate accuracy: 
```bash
python bench_*.py
```

Evaluate speed: 
```bash
python speed.py
```

### 4. Accuracy/Speed 

Performance of different datasets.

|Model         |PSNR                    |SSIM|
|:------------:|:-----------------:|:----------:|
|Vimeo90K     |35.47               |0.978       |
|UCF101       |35.28               |0.969       |
|ATD12K       |28.91               |0.957       |
|Xiph-2K (resized)  |36.44         |0.966       |
|Xiph-4K (cropped)  |33.92         |0.945       |
|XTEST-2K     |32.13               |0.926       |
|XTEST-4K     |30.88               |0.914       |



Speed (on a Titan X) for different settings.
|Input Size         |Inter Steps                    |Speed (ms/f) |
|:------------:|:-----------------------:|:-----------------:|
|480 x 640     |x 2                   |36       |
|480 x 640     |x 4                   |15       |
|480 x 640     |x 8                   |9       |
|480 x 640     |x 16                   |6       |
