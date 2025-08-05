import pandas as pd
import random

def poison_labels(df, target_col='species', poison_fraction=0.1):
    df_poisoned = df.copy()
    unique_labels = df_poisoned[target_col].unique()

    num_poison = int(len(df_poisoned) * poison_fraction)
    poison_indices = random.sample(list(df_poisoned.index), num_poison)

    for idx in poison_indices:
        original_label = df_poisoned.loc[idx, target_col]
        other_labels = [label for label in unique_labels if label != original_label]
        df_poisoned.loc[idx, target_col] = random.choice(other_labels)

    return df_poisoned

if __name__ == "__main__":
    # Load clean dataset
    df = pd.read_csv("iris.csv")

    # Poison dataset
    df_poisoned = poison_labels(df, poison_fraction=0.1)

    # Save poisoned dataset
    df_poisoned.to_csv("iris_poisoned.csv", index=False)

    print("Poisoned dataset saved to iris_poisoned.csv")
