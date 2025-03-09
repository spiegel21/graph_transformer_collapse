import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sns.set_theme()
if(torch.cuda.is_available):
     device = 'cuda'
else:
     device = 'cpu'

class Tracker():
    def __init__(self, last_layer, labels):
         self.last_layer = last_layer
         print(f'Variability Collapse Tracker initialized on layer {last_layer}')
         self.hook_last_layer()
         self.activation = None
         self.epoch_list = []
         self.s_w_list = []
         self.s_b_list = []
         self.stn_list = []
         self.ratio_list = []
         print(device)
         self.labels = labels.type(torch.int64).to(device)

    def hook_last_layer(self):
        def hook(model, i, o):
             self.activation = o.detach()
        self.last_layer.register_forward_hook(hook)

    def compute_nc1(self, epoch):
            feat = self.activation.T
            with torch.no_grad():
                
                class_means = scatter(feat, self.labels, dim=1, reduce="mean")
                expanded_class_means = torch.index_select(class_means, dim=1, index=self.labels)
                z = feat - expanded_class_means
                num_nodes = z.shape[1]

                S_W = z @ z.t()
                S_W /= num_nodes

                global_mean = torch.mean(class_means, dim=1).unsqueeze(-1)

                z = class_means - global_mean
                num_classes = class_means.shape[1]

                S_B = z @ z.t()
                S_B /= num_classes

                collapse_metric_type1 = torch.trace(S_W @ torch.linalg.pinv(S_B)) / num_classes
                collapse_metric_type2 = torch.trace(S_W)/torch.trace(S_B)

            self.s_w_list.append(np.log10(torch.trace(S_W).cpu()))
            self.s_b_list.append(np.log10(torch.trace(S_B).cpu()))
            self.stn_list.append(np.log10(collapse_metric_type1.cpu()))
            self.ratio_list.append(np.log10(collapse_metric_type2.cpu()))
            self.epoch_list.append(epoch)

    def plot_results(self, model_name):
         path = os.path.join('nc_arrays', model_name.lower().replace(' ', '_'))
         if not os.path.exists(path):
              os.makedirs(path)
         print(len(self.s_b_list))
         fn_out = os.path.join(path, 'variability_collapse.pdf')
         print(f'Plotting results in {fn_out}')
         print(self.epoch_list)
         plt.plot(self.epoch_list, self.s_w_list, label='S_W')
         plt.plot(self.epoch_list, self.s_b_list, label='S_B')
         plt.plot(self.epoch_list, self.stn_list, label='Signal-to-Noise')
         plt.plot(self.epoch_list, self.ratio_list, label='Ratio of Traces')
         plt.xlabel('Epoch')
         plt.ylabel('Quantity')
         plt.ylim(-5, 5)
         plt.title(f'{model_name} NC1 Graph')
         plt.legend()
         plt.savefig(fn_out)

         np.savetxt(os.path.join(path, 'epoch_arr.txt'), np.array(self.epoch_list))
         np.savetxt(os.path.join(path, 's_w_arr.txt'), np.array(self.s_w_list))
         np.savetxt(os.path.join(path, 's_b_arr.txt'), np.array(self.s_b_list))
         np.savetxt(os.path.join(path, 'stn_arr.txt'), np.array(self.stn_list))
         np.savetxt(os.path.join(path, 'ratio_arr.txt'), np.array(self.ratio_list))