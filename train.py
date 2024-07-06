import torch
import transformers
import numpy as np
from model.image_processing import Mask2FormerImageProcessor
from model.modeling import Mask2FormerForUniversalSegmentation
from torch.utils import data
from torch import nn
import os

from datasets import load_dataset
import torchvision.transforms as tfs

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
import argparse

def args():
    parser = argparse.ArgumentParser(description='MaskCD Training Arguments')
    parser.add_argument('--dataset', type=str, default='ericyu/CLCD_Cropped_256', help='dataset id')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
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
        return imageA,imageB,label

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
    return batch

def main(args):
    logger = get_logger(__name__)
    accelerator=Accelerator()
    batch_size=args.batch_size
    preprocessor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
    dataset=load_dataset(args.dataset)
    train_ds=dataset["train"]
    val_ds=dataset["val"]
    transform=tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=ADE_MEAN,std=ADE_STD),
    ])

    train_dataset=ChangeDetectionDataset(train_ds,transform=transform)
    val_dataset=ChangeDetectionDataset(val_ds,transform=transform)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-ade-semantic",ignore_mismatched_sizes=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-7, last_epoch=-1, verbose=False)


    model, optimizer, train_dataloader, test_dataloader, scheduler=accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler)
    
    best_f1=0
    for epoch in range(args.epochs):
        logger.info(f'Epoch:{epoch}',main_process_only=True)
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader,disable=not accelerator.is_local_main_process, miniters=20)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            
            pixel_values=batch["pixel_values"]
            mask_labels=[labels for labels in batch["mask_labels"]][0:batch_size]
            class_labels=[labels for labels in batch["class_labels"]][0:batch_size]
            outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
            # Backward propagation
            loss = outputs.loss
            accelerator.backward(loss)

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print("Loss:", running_loss/num_samples)

            # Optimization
            optimizer.step()
            scheduler.step()

        if epoch%5==0:
            TP,TN,FP,FN=0,0,0,0
            model.eval()
            for idx, batch in enumerate(tqdm(test_dataloader, disable=not accelerator.is_local_main_process, miniters=20)):
                pixel_values = batch["pixel_values"]
        
                # Forward pass
                with torch.no_grad():
                    outputs = model(pixel_values=pixel_values)

                # get original images
                original_images = batch["original_images"]
                target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
                # print(f'trg_sizes={target_sizes}')
                # predict segmentation maps
                predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                            target_sizes=target_sizes)
                predicted_segmentation_maps=torch.stack(predicted_segmentation_maps)

                labels=torch.cat(batch["original_segmentation_maps"])
                tp,fp,tn,fn=confusion(predicted_segmentation_maps,labels.squeeze())
                TP+=tp
                TN+=tn
                FP+=fp
                FN+=fn
            f1=2*TP/(2*TP+FP+FN)
            ts_metrics_list=torch.FloatTensor([f1]).cuda().unsqueeze(0)
            ts_eval_metric_gathered=accelerator.gather(ts_metrics_list)
            final_metric=torch.mean(ts_eval_metric_gathered, dim=0)
            f1=final_metric[0]

            accelerator.print(f'Epoch {epoch} finished, evaluated F1-Score:{final_metric[0]}')
            save_pretrained_path=f"./exp/{args.dataset.split('/')[-1]}/{epoch}"
            os.makedirs(save_pretrained_path,exist_ok=True)
            saved_model=accelerator.unwrap_model(model)
            saved_model.save_pretrained(save_pretrained_path)

            if f1>best_f1:
                save_pretrained_path=f"./exp/{args.dataset.split('/')[-1]}/best_f1"
                os.makedirs(save_pretrained_path,exist_ok=True)
                saved_model.save_pretrained(save_pretrained_path)
            

if __name__=="__main__":
    args=args()
    main(args)
