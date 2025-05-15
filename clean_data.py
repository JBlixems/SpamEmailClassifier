import os
import glob
import csv
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

def normalize_and_load(folder_path):
    """
    Load all CSVs in folder_path—respecting quoted commas—and
    normalize to DataFrame with columns ['text_combined', 'label'].
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    records = []
    for path in all_files:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, quotechar='"')  # handles commas inside quotes :contentReference[oaicite:3]{index=3}
            for row in reader:
                # Build combined text
                if 'text_combined' in row and row['text_combined']:
                    text = row['text_combined']
                else:
                    subj = row.get('subject', '').strip()
                    body = row.get('body', '').strip()
                    # Some files may have a 'text' column
                    if not subj and not body:
                        subj = row.get('text', '').strip()
                    text = f"{subj} {body}".strip()
                label = row.get('label')
                if text and label is not None:
                    records.append((text, label))

    # Create DataFrame
    df = pd.DataFrame(records, columns=['text_combined', 'label'])
    return df

def split_and_save(df, train_path='train.csv', test_path='test.csv',
                   test_size=0.15, random_state=42):
    """
    Shuffle, split into train/test, and save as CSV.
    """
    train_text, test_text, train_lbl, test_lbl = train_test_split(
        df['text_combined'], df['label'],
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    train_df = pd.DataFrame({'text_combined': train_text, 'label': train_lbl})
    test_df  = pd.DataFrame({'text_combined': test_text,  'label': test_lbl})

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    print(f"[✓] Training set: {len(train_df)} rows → {train_path}")
    print(f"[✓] Testing set:  {len(test_df)} rows → {test_path}")

def sanitize_commas(df, placeholder='|'):
    df = df.copy()
    df['text_combined'] = df['text_combined'].str.replace(',', placeholder)
    return df

if __name__ == "__main__":
    data = normalize_and_load("TrainingSet")
    data = sanitize_commas(data)
    print(f"Total records after merge: {len(data)}")
    split_and_save(data)
