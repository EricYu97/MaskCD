from datasets import Dataset, DatasetDict, Image
import os

# your images can of course have a different extension
# semantic segmentation maps are typically stored in the png format

train_A_path="./SYSU_CD/train/A/"
train_B_path="./SYSU_CD/train/B/"
train_label_path="./SYSU_CD/train/label/"

test_A_path="./SYSU_CD/test/A/"
test_B_path="./SYSU_CD/test/B/"
test_label_path="./SYSU_CD/test/label/"

val_A_path="./SYSU_CD/val/A/"
val_B_path="./SYSU_CD/val/B/"
val_label_path="./SYSU_CD/val/label/"

train_flist=os.listdir(train_A_path)
test_flist=os.listdir(test_A_path)
val_flist=os.listdir(val_A_path)

train_imageA_paths=[train_A_path+i for i in train_flist]
train_imageB_paths=[train_B_path+i for i in train_flist]
train_label_paths=[train_label_path+i.replace("jpg","png") for i in train_flist]

test_imageA_paths=[test_A_path+i for i in test_flist]
test_imageB_paths=[test_B_path+i for i in test_flist]
test_label_paths=[test_label_path+i.replace("jpg","png") for i in test_flist]

val_imageA_paths=[val_A_path+i for i in val_flist]
val_imageB_paths=[val_B_path+i for i in val_flist]
val_label_paths=[val_label_path+i.replace("jpg","png") for i in val_flist]



# same for validation
# image_paths_validation = [...]
# label_paths_validation = [...]

def create_dataset(imageA_paths, imageB_paths, label_paths):
    dataset = Dataset.from_dict({"imageA": sorted(imageA_paths),
                                 "imageB": sorted(imageB_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("imageA", Image())
    dataset = dataset.cast_column("imageB", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

# step 1: create Dataset objects
train_dataset=create_dataset(imageA_paths=train_imageA_paths,imageB_paths=train_imageB_paths,label_paths=train_label_paths)
test_dataset=create_dataset(imageA_paths=test_imageA_paths,imageB_paths=test_imageB_paths,label_paths=test_label_paths)
val_dataset=create_dataset(imageA_paths=val_imageA_paths,imageB_paths=val_imageB_paths,label_paths=val_label_paths)

# step 2: create DatasetDict
dataset = DatasetDict({
    "train":train_dataset,
    "test":test_dataset,
    "val":val_dataset
  }
)

# step 3: push to hub (assumes you have ran the huggingface-cli login command in a terminal/notebook)
# dataset.push_to_hub("")
dataset.save_to_disk("./SYSUCD/")