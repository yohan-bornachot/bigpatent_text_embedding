# BigPatent Text Embedding

## Train a model

### Launch the following command line and adapt arguments to your own file tree:
 > python -m train --cfg_path </path/to/config/file.yml> --data_path </path/to/data/dataset_big_patent_v1.json> --output_dir </path/to/output/dir> --device <device> --cache_dir <path/to/cache/dir>

With arguments:
- **cfg_path**: Path to config file
- **data_path**: Path to dataset file (expects a .json file)
- **output_dir**: Output directory to store results
- **device**: Device to use for computations
- **cache_dir**: Directory where to cache models when loading from HuggingFace

### Evaluate a model on test dataset:

 > python -m inference --cfg_path </path/to/config/file.yml> --data_path </path/to/data/dataset_big_patent_v1.json> --output_dir </path/to/output/dir> --device <device> --cache_dir <path/to/cache/dir>

With arguments:
- **cfg_path**: Path to config file
- **pretrained_model**: Path or name of model file.
- **data_path**: Path to dataset file (expects a .json file)
- **output_dir**: Output directory to store results
- **device**: Device to use for computations

### Results in notebook file

The notebook file contained in this repository carries out a basic data analysis, 
and compares zero-shot performances of a chosen text embedding model to fine-tuned 
performances of the same model. However, fine-tuning must be launched separately,
and fine-tuned model path should point to training output directory.