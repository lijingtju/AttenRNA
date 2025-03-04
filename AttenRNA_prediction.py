import torch
import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from imageDataLoader import ImageDataset
from resnet import resnet18AndMultiHeadAttention2Feature
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse






def loadModel(RESUME, MLP_INPUT, NUM_CLASSES):
    model = resnet18AndMultiHeadAttention2Feature(MLP_INPUT, NUM_CLASSES)
    if RESUME is not None:
        checkpoint = torch.load(RESUME, weights_only=False)
        ckp_keys = list(checkpoint['model_state_dict'])
        cur_keys = list(model.state_dict())
        model_sd = model.state_dict()
        for ckp_key in ckp_keys:
            model_sd[ckp_key] = checkpoint['model_state_dict'][ckp_key]
        model.load_state_dict(model_sd)
    return model


def setup_device(n_gpu_use=0):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids




def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--data", type=str, default=None, help="")
    parser.add_argument("--log_dir", type=str, default=None, help="")
    parser.add_argument("--batch", type=int, default=32, help="batch")
    parser.add_argument("--resume", type=str, default=None, help="The path of the pre-trained model.")

    args = parser.parse_args()

    test_data = args.data #"./mouse_RNA.csv"
    result_save_dir = args.log_dir # "./logs"
    BATCH_SIZE = args.batch # 32
    RESUME = args.resume # './mouse_model.pt'
    NUM_CLASSES = 3
    MLP_INPUT = 1344

    test_dataset = ImageDataset(dataPath=test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=10,
                                            pin_memory=True)
    test_total = len(test_dataloader.dataset)
    test_steps = len(test_dataloader)


    device,_ = setup_device()
    model = loadModel(RESUME, MLP_INPUT, NUM_CLASSES)
    model.to(device)
    testacc,  pre_result = evalate(model=model, dataLoader=test_dataloader, device=device, test_total=test_total, test_steps=test_steps, ep=0)
    print("test acc:{:.3f}%\n".format(testacc*100))
    lines = "acc:,{}\n\nlabel,pred,seq\n".format(testacc)

    lines += pre_result
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    with open(os.path.join(result_save_dir, "{}_{}.csv".format(os.path.basename(test_data).replace(".", ""), int(time.time()))), "w", encoding="utf-8") as f:
        f.write(lines)



def evalate(model, dataLoader, device, test_total, test_steps, ep):
    model.eval()
    count = 0
    pre_result = ""
    for step, data in enumerate(dataLoader):
        label_index, labels, seq, xs = data
        xs = xs.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        pred = model(xs)
        predCpu = pred.cpu()
        predCpu = torch.argmax(predCpu, dim=1)
        predCpu = predCpu.tolist()
        labelsCpu = labels.cpu().to(torch.int32)
        labelsCpu = labelsCpu.tolist()
        for i, pre in enumerate(predCpu):
            pre_result += "{},{},{}\n".format(int(label_index[i].item()), pre, seq[i])
            if pre == int(label_index[i].item()):
                count+=1
        print("\r[ test process ] {}/{}".format(step+1, test_steps), end='')
    print("\n")
    return count/test_total, pre_result

if __name__ == "__main__":
    main()