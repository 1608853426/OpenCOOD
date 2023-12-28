from opencood.communication.data_loader import *
from opencood.tools import train_utils, inference_utils
import torch
from collections import OrderedDict
from types import SimpleNamespace
class FeatureExtractor():
    
    def __init__(self) -> None:
        self.where2comm_data = Where2commData()
        self.opt = self.where2comm_data.get_opt()
        self.hypes = self.where2comm_data.hypes
        self.config = SimpleNamespace(**self.hypes)
        self.config.model['core_method'] = 'point_pillar_where2comm_cav'
        self.hypes_new = vars(self.config)
        self.model = train_utils.create_model(self.hypes_new)
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_path = self.opt.model_dir
        _, self.model = train_utils.load_saved_model(saved_path, self.model)
        self.model.eval()
    
    def extract(self, data):
        pass
    
    def extract_idx(self, idx):
        batch_data = self.where2comm_data.get_idx_data(idx)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)
            cav_content = batch_data['ego']
            output_dict = OrderedDict()
            output_dict['ego'] = self.model(cav_content)
        
