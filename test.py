import torch
import transformers
import numpy as np
from model.image_processing import Mask2FormerImageProcessor
from model.modeling import Mask2FormerForUniversalSegmentation
from torch.utils import data
from torch import nn
import os
from PIL import Image

from datasets import load_dataset, load_from_disk
import torchvision.transforms as tfs

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

import argparse

miou_list=[]
f1_list=[]

def args():
    parser = argparse.ArgumentParser(description='MaskCD Testing Arguments')
    parser.add_argument('--model', type=str, default='ericyu/CLCD_Cropped_256', help='model id')
    parser.add_argument('--dataset', type=str, default='ericyu/CLCD_Cropped_256', help='dataset id')
    args = parser.parse_args()
    return args

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
class ChangeDetectionDataset(data.Dataset):
    def __init__(self,dataset,transform=None) -> None:
        super().__init__()
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return(len(self.dataset))
    def __getitem__(self, index):
        imageA=self.transform(self.dataset[index]["imageA"])
        imageB=self.transform(self.dataset[index]["imageB"])
        label=tfs.ToTensor()(self.dataset[index]["label"])
        label=torch.cat([label],dim=0)
        return imageA,imageB,label,index

def collate_fn(batch):
    preprocessor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
    inputs=list(zip(*batch))
    imageA=inputs[0]
    imageB=inputs[1]
    label=inputs[2]
    imageAB=imageA+imageB
    label2=label+label
    batch=preprocessor(imageAB,segmentation_maps=label2,return_tensors='pt')
    batch["original_images"] = inputs[1]
    batch["original_segmentation_maps"] = inputs[2]
    batch["index"]=inputs[3]
    return batch

def main(model_name,dataset_name,model_id,num_classes):
    print(f'testing {model_name}')
    print(transformers.__file__)

    logger = get_logger(__name__)
    accelerator=Accelerator()
    device=accelerator.device
    batch_size=10
    preprocessor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    dataset=load_dataset(dataset_name)
    logger.info(dataset,main_process_only=True)
    test_ds=dataset["test"]

    transform=tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=ADE_MEAN,std=ADE_STD),
    ])


    test_dataset=ChangeDetectionDataset(test_ds,transform=transform)

    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id,ignore_mismatched_sizes=True)

    model = model.to(device)

    model, test_dataloader=accelerator.prepare(model,test_dataloader)
    model.eval()
    
    TP,TN,FP,FN=0,0,0,0 
    os.makedirs(f"./results/{model_name}/change_map/",exist_ok=True)

    accelerator.print(f'Testing {model_name} on {dataset_name}...')
    for i, batch in enumerate(tqdm(test_dataloader,disable=not accelerator.is_local_main_process, miniters=20)):
        with torch.no_grad():
            outputs=model(batch["pixel_values"].to(device))
            original_images = batch["original_images"]
            target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
            predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            img_idx=batch["index"]
            labels=torch.stack(list(batch["original_segmentation_maps"]), dim=0)
            predicted_segmentation_maps = torch.stack(predicted_segmentation_maps, dim=0)
            class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

            segmentation_256=torch.nn.functional.interpolate(segmentation, size=(256, 256), mode="bilinear", align_corners=False)

            change_map=torch.stack([segmentation_256[:,0,:,:],segmentation_256[:,1,:,:]],dim=1).softmax(dim=1)

            tp,fp,tn,fn=confusion(change_map.argmax(dim=1),labels.squeeze())
            TP+=tp
            TN+=tn
            FP+=fp
            FN+=fn

            for i in range(len(predicted_segmentation_maps)):
                segmentation_map = Image.fromarray((255*change_map[i,].argmax(dim=0)).cpu().numpy().astype(np.uint8))
                segmentation_map.save(os.path.join(f"./results/{model_name}/change_map/"+str(img_idx[i])+".png"))

    OA=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1=2*TP/(2*TP+FP+FN)
    cIoU=TP/(TP+FP+FN)
    ts_metrics_list=torch.FloatTensor([OA,f1,precision,recall, cIoU]).cuda().unsqueeze(0)
    ts_eval_metric_gathered=accelerator.gather(ts_metrics_list)
    final_metric=torch.mean(ts_eval_metric_gathered, dim=0)
    accelerator.print(f'Accuracy={final_metric[0]}, mF1={final_metric[1]}, Precision={final_metric[2]}, Recall={final_metric[3]}, cIoU={final_metric[4]}')
    # if accelerator.is_local_main_process:
    #     model = model.push_to_hub('ericyu/MaskCD_EGY_BCD')

if __name__=="__main__":
    args=args()
    main(model_name=args.model,dataset_name=args.dataset,model_id=args.model,num_classes=2)
