## Training M2M

### 1. Prepare Data
Download [Vimeo90k](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip) and save datasets. 

### 2. Prepare Environment
Run 
```bash
pip install -r requirements.txt
```

### 3. Modify Codes
Modify `*.yml` files in `./config`
* ''data:path'': path to dataset 
* ''training:batch_size'': batch_size

### 4. Training
Run
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 26666 train.py --config configs/vimeo90k.yml
```
