import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


UID_COL = 'UniprotID'
SEQ_COL = 'sequence'
RXN_COL = 'CANO_RXN_SMILES'


class BaseModel(nn.Module):
    
    @torch.no_grad()
    def evaluate(self, dataloader, show_progress=False):
        preds = []
        targets = []
        rxns = []
        
        if show_progress:
            dataloader = tqdm(dataloader, total=len(dataloader))

        device = self.device if hasattr(self, 'device') else self.model_device
        for batch in dataloader:
            pred = self.forward(batch.to(device))
            if isinstance(pred, tuple):
                pred = pred[0]
            preds.append(pred)
            targets.append(batch.y)
            rxns.extend(batch.rxn)
        pred = torch.cat(preds)
        target = torch.cat(targets)
        
        if not self.sigmoid_readout:
            pred = torch.sigmoid(pred)

        metric = self.calc_metric(pred, target, rxns)
        return pred, metric

    def calc_metric(self, preds, target, rxns=None):
        preds = preds.cpu().tolist()
        target = [int(i) for i in target.cpu().tolist()]

        if rxns is None:
            overall_auc = float(roc_auc_score(target, preds) if len(set(target)) != 1 else -1)
        else:
            df = pd.DataFrame({"pred": preds, "Label": target, "rxns": rxns})
            auc_list = []
            cnt_bad = 0
            for _, df in df.groupby('rxns'):
                if len(set(df['Label'])) == 1:
                    cnt_bad += 1
                    continue
                single_auc = float(roc_auc_score(df['Label'].values, df['pred'].values))
                auc_list.append(single_auc)
            overall_auc = float(np.mean(auc_list))
            
            if cnt_bad > 0:
                print(f'{cnt_bad} reactions have only one class!')

        metric = {
            "Accuracy": float(accuracy_score(target, np.rint(preds))),
            "AUC": overall_auc,
            "Precision": float(precision_score(target, np.rint(preds)) if len(set(target)) != 1 else -1),
            "Recall": float(recall_score(target, np.rint(preds)) if len(set(target)) != 1 else -1),
            "F1": float(f1_score(target, np.rint(preds)) if len(set(target)) != 1 else -1),
        }
        return metric
