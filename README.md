# UEFCCS
UEFCCS was developed for collision cross section (CCS) prediction

## Setting up Conda environment

Create new Conda environment using environment.yml.  

    conda env create -f environment.yml
    conda activate uefccs

## Running evaluation
Once uefccs is correctly installed, please check the location of uefccs

    pip show uefccs

copy the fine-tuned weights form https://drive.google.com/file/d/1ja3TR2ltQBPxf87CPUa1-vAyNesIpIvg/view?usp=drive_link to the subdir "uefccs/lib/python3.10/site-packages/unimol_tools/weights/"
    
    python UEFCCS_predict.py --input_csv /test.csv --output_csv /predicts.csv
    
## reproduce  

please  refer UEFCCS architecture
1. to get the [CLS] token using 

## Citation

If you use this code, please cite:

    @article{ypsh1_2026UEFCCS,
          title={}, 
          author={},
          journal={},
          year={2026}
    }

## License
MIT
