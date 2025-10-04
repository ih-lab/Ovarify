import datetime
import os
import torch
from utils import *
import pandas as pd
from collections import defaultdict
import loss
from tqdm import tqdm


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, dataloaders, device, save_path, loss_type, start_time, num_epochs=25):
    best_loss = 1e10
    metrics_total = pd.DataFrame()

    if loss_type == 'bce':
        pos_weight = compute_class_weights(dataloaders, device, model_type='class')
        print("pos_weight: {}".format(pos_weight))
    else:
        print("Skipping pos_weight calculation; loss_type is not bce")

    print("---Begin training---")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = datetime.datetime.now()

        for phase in ['train', 'val']:
            if phase == 'train':
                print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for i,data in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                if i==0:
                    tqdm.write("Phase: {}".format(phase))
                inputs, labels, classif, dirs, files, row_img, col_img = data[0].to(device), data[1].to(device), data[2].to(device), data[3], data[4], data[5], data[6]
                labels_sdm = labels.clone()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss_value = loss.calc_loss(outputs, labels, labels_sdm, classif, metrics, loss_type=loss_type, bce_weight=pos_weight if loss_type=='bce' else None, model_type='class')

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                            scheduler.step()
                        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            scheduler.step(epoch + i / len(dataloaders[phase]))
                        
                epoch_samples += inputs.size(0)

            print("Phase samples: {}".format(epoch_samples))
            epoch_loss = metrics[loss_type]/epoch_samples

            metrics['bce'] = metrics['bce']/epoch_samples
            metrics['epoch'] = epoch + 1
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']

            metrics_single = pd.DataFrame(metrics, index=[0])
            metrics_single['phase'] = phase
            wall_time = datetime.datetime.now() - start_time
            wall_time = convert_time(wall_time)
            metrics_single['cum_wall_time_training'] = wall_time
            print(metrics_single)
            metrics_total = pd.concat([metrics_total, metrics_single], axis=0, ignore_index=True)

            if phase == 'val' and epoch_loss < best_loss:
                print("---Saving best model---")
                best_loss = epoch_loss

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), os.path.join(save_path, f"weightsonly_{epoch}.trained.pth"))
                torch.save({
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None, 
                }, os.path.join(save_path, f"checkpoint_{epoch}.trained.tar"))
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

        time_elapsed = datetime.datetime.now() - since
        time_elapsed = convert_time(time_elapsed)
        print('Epoch time elapsed: {}'.format(time_elapsed))
        print('Epoch loss: {:4f}'.format(epoch_loss))
        print('Best val loss: {:4f}'.format(best_loss))
        metrics_total.to_csv(os.path.join(save_path, "metrics.csv"), mode='w')

    return metrics_total