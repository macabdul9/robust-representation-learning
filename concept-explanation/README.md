### Experiments


- Models: BERT, DistilBERT, RoBERTa, ALBERT, XLNET, etc
- Datasets: {
    0:{"IID":IMDB, "OOD":SST2},
    1:{"IID":MNLI, "OOD":HANS},
    2:{"IID":QQP, "OOD":PAWS}
}

- Objective: Train a model on IID data for each task and check the confidence of the model on IID and OOD predictions. What do low confidence and high confidence examples share. 

