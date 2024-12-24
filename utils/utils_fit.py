import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, focal_loss, Boudaryloss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
from utils.dataloader import augmentationimage as ugmentationimage
from utils.TI_loss import TI_Loss
import torch.nn.functional as F
import numpy as np



def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, loss_fuc, num_classes, save_dir, no_improve_count):
    total_loss = 0
    val_loss = 0
    val_f_score = 0

    # print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, overpngs, nonoverpng = batch

        with torch.no_grad():
            imgs = imgs.cuda()  # [bsz, 3, 448, 448]
            pngs = pngs.cuda()  # tragets
            overpngs = overpngs.cuda()  # tragets
            nonoverpngs = nonoverpng.cuda()  # tragets

        optimizer.zero_grad()
        outputs, overoutputs, nonoutputs, preoutputs = model_train(imgs)  # [bsz, 24, 448, 448]

        # ----------------------------------
        # choose the loss fuc
        # ----------------------------------
        if loss_fuc == "BCEloss":
            loss = CE_Loss(outputs, pngs)


        elif loss_fuc == "Diceloss":
            aloss = Dice_loss(outputs, pngs)
            overloss = Dice_loss(overoutputs, overpngs)
            nonoverloss = Dice_loss(nonoutputs, nonoverpngs)
            preloss = Dice_loss(preoutputs, pngs)
            conloss = mse_loss(outputs,preoutputs)
            loss = (aloss + overloss + nonoverloss + preloss ) / 4  + 0.0002 * conloss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # total_f_score   += _f_score.item()

        pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step:
            break
        imgs, pngs, overpngs, nonoverpng = batch

        with torch.no_grad():
            imgs = imgs.cuda()  # [bsz, 3, 448, 448]
            pngs = pngs.cuda()  # tragets
            overpngs = overpngs.cuda()  # tragets
            nonoverpngs = nonoverpng.cuda()  # tragets

        optimizer.zero_grad()
        outputs, overoutputs, nonoutputs,preoutputs = model_train(imgs)  # [bsz, 24, 448, 448]

        # ----------------------------------
        # choose the loss fuc
        # ----------------------------------
        if loss_fuc == "BCEloss":
            loss = CE_Loss(outputs, pngs)


        elif loss_fuc == "Diceloss":
            aloss = Dice_loss(outputs, pngs)
            overloss = Dice_loss(overoutputs, overpngs)
            nonoverloss = Dice_loss(nonoutputs, nonoverpngs)
            preloss = Dice_loss(preoutputs, pngs)
            conloss = mse_loss(outputs,preoutputs)
            loss = (aloss + overloss + nonoverloss + preloss ) / 4  + 0.0001 * conloss

        val_loss += loss.item()
        # val_f_score += _f_score.item()

        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1), 'f_score': val_f_score / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    # eval_callback.on_epoch_end(epoch + 1, model_train)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

    if epoch > 100 and epoch % 50 == 0:

        save_path = os.path.join(save_dir, f"epoch_{epoch}_weights.pth")


        torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    no_improve_count += 1
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        no_improve_count = 0

    return no_improve_count