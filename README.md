# UEFCCS
UEFCCS was developed for collision cross section (CCS) prediction

## Setting up Conda environment

Create new Conda environment using environment.yml.  

    conda env create -f environment.yml
    conda activate uefccs

## Running evaluation
once uefccs is correctly installed, please load the weights “”

    pip show uefccs

copy the weights “ckpt.pt” replace the /envs/unim/lib/python3.10/site-packages/unimol_tools/weights/mol_pre_all_h_220816.pt

    
    python UEFCCS_predict.py
    
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
