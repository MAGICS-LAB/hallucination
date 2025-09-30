# Are Hallucinations Bad Estimations?

This is the code for the paper [**Are Hallucinations Bad Estimations?**](https://arxiv.org/abs/2509.21473) You can use this repo to reproduce the results in the paper.

## Environment Setup

## Usage
1. Clone the repository
    ```bash
    gh repo clone MAGICS-LAB/hallucination
    cd hallu
    ```
2. Create and activate a virtual environment
    ```bash
    conda create -n hallu python=3.11
    conda activate hallu
    ```
3. Install required packages
    ```bash
    pip install -r requirements.txt
    ```

### Synthetic Coin Flipping Problem
Please refer to ```coinflip_experiment.ipynb```.

### Open-Ended Text Questions
```bash
python hallucination_llm.py
```

### Open-Ended Text-to-Image

1. Fine-tune Unet
    ```bash
    python text_to_img_exp/fine-tune_text_to_img.py
        -- output_dir "./model"
    ```
2. Hallucinatino Rate Analysis with Frozen Model
    ```bash
    python text_to_img_exp/hallucination_text_to_img_report.py                  
        --checkpoints_glob "./model/checkpoint-*/unet"
    ```

### Plotting
```bash
python plotting_llm.py # for plotting of Figure 4
python plotting_text_to_img.py # for plotting of Figure 5
python ablation_plotting_text_to_img.py # for ablation study plots
```

## Citation
If you find our work useful, please cite:
