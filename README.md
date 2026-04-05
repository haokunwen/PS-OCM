# Partially Supervised Compatibility Modeling [IEEE TIP 2022]


## Authors

**Weili Guan**<sup>1</sup>, **Haokun Wen**<sup>2</sup>, **Xuemeng Song**<sup>2</sup>\*, **Chun Wang**<sup>2</sup>, **Chung-Hsing Yeh**<sup>1</sup>, **Xiaojun Chang**<sup>3</sup>\*, **Liqiang Nie**<sup>4</sup>

<sup>1</sup> Monash University, Melbourne, VIC, Australia  
<sup>2</sup> Shandong University, Qingdao, China  
<sup>3</sup> University of Technology Sydney, Sydney, NSW, Australia  
<sup>4</sup> Harbin Institute of Technology (Shenzhen), Shenzhen, China  
\* Corresponding authors

## Links

- **Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/document/9817021)
- **Checkpoint**: [ps-ocm](https://drive.google.com/file/d/1-8of-n3dtLZLKy8w7ylp-QyXJDyoCWRP/view?usp=sharing)

---

## Running Environments

- Python 3.8.3
- PyTorch 1.6.0
- GPU: NVIDIA GeForce RTX 2080
- OS: Ubuntu 18.04.5 LTS

---

## Repository Structure

```text
PS-OCM/
├── Comp/               # Source code and pre-trained parameters for PS-OCM
├── FITB/               # Test code for the fill-in-the-blank task
├── data/               # Required data files
│   ├── train_list.csv
│   ├── valid_list.csv
│   ├── test_list.csv
│   ├── test_fitb_p.csv
│   ├── test_fitb_n1.csv
│   ├── test_fitb_n2.csv
│   ├── test_fitb_n3.csv
│   └── item_img_num.csv
├── partial_mask.npy    # Partial supervision mask
└── README
```

---

## Dataset Preparation

This project uses the **IQON3000** dataset. Make sure the dataset is available locally and note its path for use with the `--imgpath` argument.

The `data/` folder contains the following pre-processed files:

- `train/valid/test_list.csv` — training, validation, and test splits for the outfit compatibility estimation task.
- `test_fitb_p/n1/n2/n3.csv` — test files for the fill-in-the-blank task (positive and negative candidates).
- `item_img_num.csv` — attribute information derived from statistics on the IQON3000 dataset.

---

## Usage

### Compatibility Estimation

Source code and pre-trained parameters are located in the `./Comp` folder.

```bash
python train.py --batch_size 16 --epoch_num 100 --imgpath <iqon3000_data_path>
```

Replace `<iqon3000_data_path>` with the local path to your IQON3000 image directory.

---

### Fill-in-the-Blank

Test code is located in the `./FITB` folder.

```bash
python compute_fitb.py --imgpath <iqon3000_data_path>
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{guan2022partially,
  title   = {Partially Supervised Compatibility Modeling},
  author  = {Guan, Weili and Wen, Haokun and Song, Xuemeng and Wang, Chun and Yeh, Chung-Hsing and Chang, Xiaojun and Nie, Liqiang},
  journal = {IEEE Transactions on Image Processing},
  volume  = {31},
  pages   = {4733--4745},
  year    = {2022},
  doi     = {10.1109/TIP.2022.3187290}
}
```
