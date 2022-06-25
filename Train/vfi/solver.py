import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import zipfile
import glob
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Solver:
    def __init__(self, model, optimizer=None, loss=None, local_rank=-1):
        super(Solver, self).__init__()
    
        self.local_rank = local_rank

        self.netNetwork = model  
        self.netNetwork.to(device)
        self.netNetwork = DDP(self.netNetwork, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=False)


        self.optimG = optimizer
        self.losses = loss
    #end

    def update(self, im0, im1, gt, step=0, t_step=0, training=True, fltTimes=[0.5]):

        if training:
            self.netNetwork.train()
            self.optimG.zero_grad()

            pred = self.netNetwork(im0, im1, fltTimes)[0]

            output ={'pred':pred}

            loss_G = 0.0

            for loss_f in self.losses:
                loss_v = loss_f[2](pred,gt).mean()
                loss_G += loss_v*loss_f[1]
                output[loss_f[0]] = loss_v
            #end

            output['lossttl'] = loss_G
            loss_G.backward()

            nn.utils.clip_grad_norm_(self.netNetwork.parameters(), 1)

            self.optimG.step()
        else:
            self.netNetwork.eval()

            pred = self.netNetwork(im0, im1, fltTimes)[0]
            #pdb.set_trace()
            output ={'pred':pred}
            
            loss_G = 0.0
            for loss_f in self.losses:
                loss_v = loss_f[2](pred,gt).mean()
                loss_G += loss_v*loss_f[1]
                output[loss_f[0]] = loss_v
            #end
            output['lossttl'] = loss_G
        #end
            
        return output
    #end


    def save_model(self, path, rank, bestflag=False):
        if rank == 0:

            torch.save(self.netNetwork.module.state_dict(), '{}/model.pkl'.format(path))

            with zipfile.ZipFile('{}/code.zip'.format(path), 'w', zipfile.ZIP_STORED) as objZip:
                for strFile in sorted(glob.glob('./vfi/*')):
                    objZip.write(strFile, strFile)
                #end
            #end

            if bestflag:
                torch.save(self.netNetwork.module.state_dict(), '{}/model_best.pkl'.format(path))
            #end
        #end
    #end
    
 
    def load_model(self, path, bestflag=False, rank=0):
        if rank <= 0:
            if bestflag:
                state = torch.load(path+ '/model_best.pkl')
            else:
                state = torch.load(path+ '/model.pkl')
            #end

            self.netNetwork.load_state_dict(state) if hasattr(self.netNetwork, 'module') == False else self.netNetwork.module.load_state_dict(state)
        #end
    #end
#end
