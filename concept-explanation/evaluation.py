# import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm


def evaluate(model, loader, device):

    true_label = []
    pred_label = []
    pred_probs = []

    model.eval()
    model = model.to(device)

    for batch in tqdm(loader):

        outputs, _ = model(batch['input_ids'].to(device=device), batch['attention_mask'].to(device=device))

        pred_label += outputs.softmax(dim=-1).argmax(dim=-1).cpu().detach().tolist()
        true_label += batch['label'].cpu().tolist()

        softmax_outputs = outputs.softmax(dim=-1) # [batch, num_classes]
        gt  = batch['label'].cpu().tolist()

        # predicted probability
        probs = []
        for i in range(softmax_outputs.shape[0]):
            probs.append(softmax_outputs[i][gt[i]].item())

        pred_probs += probs

            

        # limit to evaluate only on one batch comment it before final run
        # break
    
    f1 = f1_score(y_true=true_label, y_pred=pred_label, average='macro')
    accuracy = accuracy_score(y_true=true_label, y_pred=pred_label)
    cr = classification_report(y_true=true_label, y_pred=pred_label, digits=4)

    return f1, accuracy, cr, true_label, pred_label, pred_probs