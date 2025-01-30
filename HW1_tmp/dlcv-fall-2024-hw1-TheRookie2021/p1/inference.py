import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from utils_class import  Test_Dataset, FinetuneResNet
import csv
# """"""""""""""""""""""""""""""""""""""""" Setting Config """"""""""""""""""""""""""""""""""""""""""""""""""""
# $1: path to the images csv file (e.g., hw1_hiddendata/p1_data/office/test.csv)
# $2: path to the folder containing images (e.g., hw1_hiddendata/p1_data/office/test/)
# $3: path of output .csv file (predicted labels) (e.g., output_p1/test_pred.csv)
VAL_TRANSFORM= transforms.Compose([
            transforms.Resize(128),                            
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                  std =[0.229,0.224,0.225] )
        ]) # by hw1 constrain
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='HW1 deep learning network.')
    parser.add_argument(
        '--load_dataset_root', default='./hw1_data/p1_data/office', type=str, help='')
    parser.add_argument(
        '--load_dataset_csv', default='', type=str, help='')
    parser.add_argument(
        '--load_model', default='', type=str, help='')
    
    parser.add_argument(
        '--batch_size', default=4, type=int, help='')
    
    parser.add_argument(
        '--save_csv_result', default='./p1/record', type=str, help='')

    parser.add_argument(
        '--seed', default=58, type=int, help='')
    
    return parser.parse_args()

def inference(model, device, testset_loader, result_log):
    print("=== Test Phase ===")
    test_loss = 0
    correct = 0
    result=[]
    model.eval()  # Important: set evaluation mode
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target, filename in (testset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            result.append([zip(list(filename), pred.tolist(),target.tolist())])
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(pred)
            # print(filename)
            # print(target)

    result_log.append('Test set: Average loss: {:.8f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    # print(result_log[-1]) 
    return result, correct / len(testset_loader.dataset)

if __name__ == '__main__':
  args=parse_args()
  
  """ Use GPU if available """
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  # print('Device used:', device)

  """ Loading setting """
  load_dataset_root=args.load_dataset_root
  load_dataset_csv=args.load_dataset_csv
  load_model=args.load_model
  """ Saving setting """
  save_csv_result=args.save_csv_result
  
  """ Other setting """
  batch_size=args.batch_size

  """ Validation Transform setting """
  VAL_TRANSFORM= transforms.Compose([
            transforms.Resize(128),                            
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                  std =[0.229,0.224,0.225] )
        ]) # by hw1 constrain
  
  # step: create csv file   
  with open(save_csv_result, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['id','filename','label'])

  # step: LOAD the model
  model=FinetuneResNet().to(device)
  model.load_state_dict(torch.load(load_model, weights_only=True)['state_dict'])

  # step: setup train/test dataset and dataloader 
  test_dataset=Test_Dataset(annotations_file=load_dataset_csv, transform=VAL_TRANSFORM, img_dir=load_dataset_root )
  test_dataloader=DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
#   print(len(test_dataset))
#   print(len(test_dataloader.dataset))

  result, acc=inference(model,device,test_dataloader,[])
  print(f"result: {acc}")
  with open(save_csv_result, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    id=0
    for batch in result:
       for data in batch:
          for  i in list(data):
              csv_writer.writerow([id,i[0],i[1][0]])
              id+=1

  