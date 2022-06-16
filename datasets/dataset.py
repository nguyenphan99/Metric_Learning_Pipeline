import cv2
import torch


class MVB():
    
    def __init__(self, df, transform = None):
        
        self.df = df.reset_index()
        self.transform = transform
                
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,index):
        
        img_path, class_id = self.df.loc[index, 'path'], self.df.loc[index,'class_id']
        sample = cv2.imread(img_path)
        if sample is None:
            print(img_path)
            
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]

        return sample, torch.tensor(class_id)