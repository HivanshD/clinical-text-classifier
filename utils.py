
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"text","label"} <= set(df.columns), "CSV must have text,label columns"
    return df

def save_confusion_matrix(cm, classes, out_path: str, title: str="Confusion Matrix"):
    fig = plt.figure(figsize=(4.5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
