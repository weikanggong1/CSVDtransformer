# Python code for the CSVDtransformer:
```
Weikang Gong et. al. "A foundation model for mapping the phenomic and genetic landscape of cerebral small vessel disease biomarkers", MedRxiv, 2025.
```

## Requirements
The code should work fine in multiple 3.x python environments. The pytorch 2.x+ and nibabel modules should be installed before running it.

## Usage
1. Download the example data and model checkpoints from this [link](https://drive.google.com/file/d/1Qfu9ppTrKOr6mD51C5p9jIpN-Y9rZIa3/view?usp=drive_link). Unzip it in the same directory as the apply_model.py script.
2. Please cd to the directory, and run the following command to perform the inference.
```
python apply_model.py
```
## How to prepare your data
See example_data folder downloaded and follow its structure. Specifically, every subject is in a unique folder, and is assumed to have 3 modalities available (T1w, T2FLAIR and SWI), and the brain images are registered to 1mm MNI152 standard space (Matrix dimension 182x218x182). We name them as "T1_brain_1mm_stdspace.nii.gz", "T2_brain_1mm_stdspace.nii.gz" and "SWI_brain_1mm_stdspace.nii.gz". 

```
example_data/
├── subject_01/
│   ├── T1_brain_1mm_stdspace.nii.gz
│   ├── T2_brain_1mm_stdspace.nii.gz
│   └── SWI_brain_1mm_stdspace.nii.gz
├── subject_02/
│   ├── T1_brain_1mm_stdspace.nii.gz
│   └── ...
└── ...
```

## Basic image preprocessing
The brain extraction and image registration should be performed by the user before inference. You can use the [fsl_anat](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/fsl_anat.html) function for T1w processing, and nonliearly warp it to the standard space. The T2FLAIR and SWI can be linearly registered to T1w first by [FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/index.html), and then nonlinearly warp to the standard space. Alternatively, you can use Freesurfer to do these steps separelately for each modality by [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) and [SynthMorph](https://martinos.org/malte/synthmorph/).

## Output
The output is a csv files. For each row, the first 3 columns are the T1w, T2FLAIR and SWI files names of a subject. The next 6 columns are the 6 CSVD biomarkers as proposed in the paper, i.e., PWNH (0-3), DWMH(0-3), Fazekas score(0-6), EPVS(0-3), LI(0-1), CMB(0-1).
