# Python code implementing the CSVDtransformer:
```
Weikang Gong et. al "A foundation model for mapping the phenomic and genetic landscape of cerebral small vessel disease biomarkers" Medrxiv, 2025.
```
## Usage
1. Download the example data and model check points from this [link](https://github.com/weikanggong1/CSVDtransformer). Put them in the same directory as the apply_model.py script.
2. Please cd to the directory, and run the following code to perform the inference.
```
python apply_model.py
```
## How to prepare your data
See example_data folder downloaded and follow its structure. Specifically, every subject is in a unique folder, and is assumed to have 3 modalities available (T1w, T2FLAIR and SWI), and the brain images are registered to 1mm MNI152 standard space (Matrix dimension 182x218x182). We name them as "T1_brain_1mm_stdspace.nii.gz", "T2_brain_1mm_stdspace.nii.gz" and "SWI_brain_1mm_stdspace.nii.gz".

## Output
The output is a csv files. For each row, the first 3 columns are the T1w, T2FLAIR and SWI files names of a subject. The next 6 columns are the 6 CSVD biomarkers as proposed in the paper.
