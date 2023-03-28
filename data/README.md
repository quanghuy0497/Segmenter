## Instruction for training with a custom dataset

This is an instruction to work with Segmenter on a custom dataset. It is quite painful to make it (I know, as the Segmenter's authors want us to suffer). Here is what I have done with the StreetHazards dataset, you should replace with another custom dataset on your own:

1. Download the dataset and stored on `data/`. For the convenient, the structure of StreetHazards is like this without any subfolder:

       ├ StreetHazards
       |    ├ annotations/
       |    |    ├ test/
       |    |    ├ training/
       |    |    └ validation/
       |    ├ images/
       |    |    ├ test/
       |    |    ├ training/
       |    |    └ validation/

2. Go to `data/config/`, then create 2 file: `streethazards.yml` and `streethazards.py` like I have made:
   1. `streethazards.yml`: this file is very simple, just follow severals samples in this repository
   2. `streethazards.py`: please carefully set the `dataset_type`, `data_root`, train/val/test path for image file and annotation ground-truth. You may also want to modify augmentation parameters a little bit to fit with your data
3. Go to `data/`, then create another file `streethazards.py`. Modify all the configuration path (line 9, 10, 23) to the dataset path in **step 1** and 2 created files in **step 2**. Set the class name to be same as `dataset_type`
4.  Go to `data/__init__.py`, add the customized class you created in **step 3**, add dataset option on `data/factory.py` (just like other examples). We have the final result like this:

        ├── data
        │   ├── streethazards.py (StreetHazardsDataset)  # step 3
        │   ├── streethazards    (dataset images)        # step 1
        │   ├── config
        |        ├── streethazard.yml                    # step 2 
        |        ├── streethazard.py                     # step 2
        │   ├── factory.py (add choices)                 # step 4
        │   ├── __init__ (addd customized class)         # step 4
5. Go to `data/mmseg_config/`, creaty a `streethazard.py` file like I do in the repository (you can seek for other examples ade20k or cityscapes)
   1. Remember to use the class name similar to the one you set in `dataset_type` (**step 2**). 
   2. Modify `CLASSES` and `PALETTE` based on what you have done in `.yml` file as in step 2. Also change the `img_suffix` and `seg_map_suffix` (line 34, 35) to match with your dataset
6. Move this created `streethazard.py` file in **step 5** to `/home/<username>/miniconda3/envs/<env_name>/lib/python3.7/site-packages/mmseg/datasets/`. Go to `__init__.py` and add customized class based on the file you just add to the folder just like other dataset
7. go to `config.yml` on main folder and add configuration on your new custom dataset. See the examples from the previous dataset

Now the setup for a new custom dataset is finished (damn, it is hard`n). You should try to run segmenter with the new dataset it to see if there is any error.

Another instruction here: https://github.com/rstrudel/segmenter/issues/56

### Download the StreedHazards dataset
 
+ Train: https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar  
+ Test:  https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar
+ Citation:

    @article{hendrycks2019anomalyseg,
      title={A Benchmark for Anomaly Segmentation},
      author={Hendrycks, Dan and Basart, Steven and Mazeika, Mantas and Mostajabi, Mohammadreza and Steinhardt, Jacob and Song, Dawn},
      journal={arXiv preprint arXiv:},
      year={2019}
    }