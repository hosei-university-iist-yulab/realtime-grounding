# Dataset Download Instructions

This document provides instructions for downloading and setting up the datasets used in the TGP (Temporal Grounding Pipeline) experiments.

## Available Datasets

| Dataset | Size | License | Status |
|---------|------|---------|--------|
| BDG2 (Building Data Genome 2) | ~10 GB | CC-BY-4.0 | Included |
| REDD (Reference Energy Disaggregation Dataset) | ~3 GB | Academic | Manual download |
| UK-DALE (UK Domestic Appliance-Level Electricity) | ~20 GB | CC-BY-4.0 | Manual download |

## 1. BDG2 Dataset (Included)

The Building Data Genome Project 2 dataset is already configured in `data/raw/bdg2/`.

**Source**: https://github.com/buds-lab/building-data-genome-project-2

No additional action required.

## 2. REDD Dataset

### Download Instructions

1. **Register** at http://redd.csail.mit.edu/ to get access

2. **Download** the low-frequency data:
   ```bash
   cd data/raw
   mkdir -p redd
   cd redd

   # Download after receiving access link
   wget [YOUR_ACCESS_LINK]/low_freq.tar.bz2
   tar -xjf low_freq.tar.bz2
   ```

3. **Expected structure**:
   ```
   data/raw/redd/
   ├── house_1/
   │   ├── channel_1.dat
   │   ├── channel_2.dat
   │   └── ...
   ├── house_2/
   └── ...
   ```

### Data Format

- Each `.dat` file contains two columns: `timestamp power_reading`
- Timestamps are Unix timestamps
- Power readings are in watts

### Citation

```bibtex
@inproceedings{kolter2011redd,
  title={REDD: A public data set for energy disaggregation research},
  author={Kolter, J Zico and Johnson, Matthew J},
  booktitle={Workshop on data mining applications in sustainability (SIGKDD)},
  year={2011}
}
```

## 3. UK-DALE Dataset

### Download Instructions

1. **Download** from https://jack-kelly.com/data/

   ```bash
   cd data/raw
   mkdir -p ukdale
   cd ukdale

   # Download house files
   wget https://data.ukedc.rl.ac.uk/simplebrowse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip
   unzip ukdale.h5.zip
   ```

2. **Alternative**: Use NILMTK-compatible format
   ```python
   from nilmtk import DataSet
   dataset = DataSet('/path/to/ukdale.h5')
   ```

3. **Expected structure**:
   ```
   data/raw/ukdale/
   ├── ukdale.h5
   └── metadata/
       ├── building1.yaml
       ├── building2.yaml
       └── ...
   ```

### Data Format

- HDF5 format with hierarchical structure
- Contains 5 houses with appliance-level data
- Sampling rates vary (6s for main, 1-6s for appliances)

### Citation

```bibtex
@article{kelly2015uk,
  title={The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes},
  author={Kelly, Jack and Knottenbelt, William},
  journal={Scientific data},
  volume={2},
  number={1},
  pages={1--14},
  year={2015}
}
```

## Verification

After downloading, verify the datasets are accessible:

```bash
python -c "
from src.data.loaders import REDDLoader, UKDALELoader

# Check REDD
try:
    redd = REDDLoader('data/raw/redd')
    print(f'REDD: {len(redd.list_buildings())} buildings')
except Exception as e:
    print(f'REDD not available: {e}')

# Check UK-DALE
try:
    ukdale = UKDALELoader('data/raw/ukdale')
    print(f'UK-DALE: {len(ukdale.list_buildings())} buildings')
except Exception as e:
    print(f'UK-DALE not available: {e}')
"
```

## Fallback Behavior

If real datasets are not available, experiments will use **simulated data** based on the BDG2 patterns. This is clearly marked in the results with `"simulated": true`.

For publication-quality results, download the actual datasets.

## Storage Requirements

| Dataset | Compressed | Extracted |
|---------|------------|-----------|
| BDG2 | N/A | ~10 GB |
| REDD | ~1 GB | ~3 GB |
| UK-DALE | ~7 GB | ~20 GB |

**Total**: ~33 GB recommended free space

## Notes

- All datasets are used for **research purposes only**
- Ensure compliance with each dataset's license before use
- For cross-dataset validation, at least BDG2 + one additional dataset is recommended
