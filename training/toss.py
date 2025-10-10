import torch
import torch.nn as nn
import torch.nn.functional as F

def ttt_loss_func(ttt_preds, ttt_gt, norm=1,
                  gamma=0.9, 
                  lambda_ttt=1.3, lambda_cok=1.0, lambda_cko=5.0,
                  **kwargs,):
    
    n_predictions = len(ttt_preds)
    #assert n_predictions == 2, "Incorrect number of predictions"
    full_loss = 0.0
    loss_ttt = 0.0
    loss_cok = 0.0
    loss_cko = 0.0

    # exlude invalid areas
    valid = ttt_gt[:,1,:,:] > 0.5 #to transform the tensor in a boolean tensor

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        if norm == 1:
            i_loss_ttt = (ttt_preds[i][:, 0, :, :] - ttt_gt[:,0,:,:]).abs()
        elif norm == 2:
            #i_loss_ttt = ((ttt_preds[i][:, 0, :, :] - ttt_gt[:,0,:,:])**2).sqrt()
            i_loss_ttt = torch.norm((ttt_preds[i][:, 0, :, :] - ttt_gt[:,0,:,:]).unsqueeze(1), p=2, dim=1)
        else:
            assert False, "Wrong norm choice. Norm must be 1 or 2"
        i_loss_ttt = (valid * i_loss_ttt).sum()
        loss_ttt += (lambda_ttt * i_loss_ttt)

        if norm == 1:
            i_loss_conf = (ttt_preds[i][:, 1, :, :] - ttt_gt[:,1,:,:]).abs()
        elif norm == 2:
            #i_loss_conf = ((ttt_preds[i][:, 1, :, :] - ttt_gt[:,1,:,:])**2).sqrt()
            i_loss_conf = torch.norm((ttt_preds[i][:, 1, :, :] - ttt_gt[:,1,:,:]).unsqueeze(1), p=2, dim=1)
        else:
            assert False, "Wrong norm choice. Norm must be 1 or 2"
        i_loss_cok = (valid * i_loss_conf).sum()
        i_loss_cko = ((~valid) * i_loss_conf).sum()

        loss_cok += (lambda_cok * i_loss_cok)
        loss_cko += (lambda_cko * i_loss_cko)
        
        i_full_loss = (lambda_ttt*i_loss_ttt) + (lambda_cok*i_loss_cok) + (lambda_cko*i_loss_cko)
        full_loss += (i_weight * i_full_loss)
        #full_loss += (lambda_ttt*i_loss_ttt)
        
    if valid.max() < 0.5:
        pass

    metrics = {
        'ttt': loss_ttt.item(),
        'cok': loss_cok.item(),
        'cko': loss_cko.item(),
    }

    return full_loss, metrics


def ttt_loss_func_general(ttt_pred, ttt_gt, norm=1,
                  lambda_ttt=1.3, lambda_cok=1.0, lambda_cko=5.0,
                  **kwargs,):
    
    full_loss = 0.0
    loss_ttt = 0.0
    loss_cok = 0.0
    loss_cko = 0.0
    
    # exlude invalid areas
    valid = ttt_gt[:,1,:,:] > 0.5 #to transform the tensor in a boolean tensor

    # ttt prediction loss ###################################
    if norm == 1:
        loss_ttt = (ttt_pred[:, 0, :, :] - ttt_gt[:,0,:,:]).abs()
    elif norm == 2:
        #loss_ttt = ((ttt_pred[:, 0, :, :] - ttt_gt[:,0,:,:])**2).sqrt()
        loss_ttt = torch.norm((ttt_pred[:, 0, :, :] - ttt_gt[:,0,:,:]).unsqueeze(1), p=2, dim=1)
        #print(ttt_pred[:, 0, :, :])
        #print(ttt_gt[:,0,:,:])
        #print((ttt_pred[:, 0, :, :] - ttt_gt[:,0,:,:])**2)
        #print(((ttt_pred[:, 0, :, :] - ttt_gt[:,0,:,:])**2).sqrt())
        #assert False
    else:
        assert False, "Wrong norm choice. Norm must be 1 or 2"
        
    loss_ttt = (valid * loss_ttt).sum()
    loss_ttt = (lambda_ttt * loss_ttt)
        
    ##########################################################
#
    ## confidence loss #######################################
    if norm == 1:
        loss_conf = (ttt_pred[:, 1, :, :] - ttt_gt[:,1,:,:]).abs()
    elif norm == 2:
        #loss_conf = ((ttt_pred[:, 1, :, :] - ttt_gt[:,1,:,:])**2).sqrt()
        loss_conf = torch.norm((ttt_pred[:, 1, :, :] - ttt_gt[:,1,:,:]).unsqueeze(1), p=2, dim=1)
    else:
        assert False, "Wrong norm choice. Norm must be 1 or 2"
    ##loss_conf = (loss_conf**2).sqrt()
    loss_cok = (valid * loss_conf).sum()
    loss_cko = ((~valid) * loss_conf).sum()
    
    loss_cok = (lambda_cok * loss_cok)
    loss_cko = (lambda_cko * loss_cko)
    ########################################################

    # full loss
    full_loss = loss_ttt + loss_cok + loss_cko

    if torch.isnan(full_loss):
        print(ttt_pred[:, 0, :, :])
        print(ttt_gt[:,0,:,:])
        print((ttt_pred[:, 0, :, :] - ttt_gt[:,0,:,:])**2)
        print(((ttt_pred[:, 0, :, :] - ttt_gt[:,0,:,:])**2).sqrt())
        assert False
    
        
    ###if valid.max() < 0.5:
    ###    pass
    #loss_cok = loss_cok
    #loss_cko = loss_cko
    metrics = {
        'ttt': loss_ttt.item(),
        'cok': loss_cok.item(),
        'cko': loss_cko.item(),
    }

    #return full_loss, metrics
    return full_loss, metrics
