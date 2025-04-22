import torch
import nltk
import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from datasets import load_dataset, concatenate_datasets, Dataset
import torch.multiprocessing as mp
import random
import ssl
import math # For ceiling function

# --- Start: Add this if you encounter SSL certificate issues during download ---
# (SSL context setting code as before)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- End: SSL Fix ---

# --- NLTK Data Download ---
def download_nltk_data():
    # (NLTK download code as before)
    try:
        nltk.data.find('corpora/wordnet')
        print("WordNet already downloaded.")
    except LookupError:
        print("WordNet not found. Downloading...")
        try:
            nltk.download('wordnet', quiet=True)
            print("WordNet downloaded successfully.")
        except Exception as e:
            print(f"Error downloading WordNet: {e}")

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
        print("Averaged Perceptron Tagger already downloaded.")
    except LookupError:
        print("Averaged Perceptron Tagger not found. Downloading...")
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            print("Averaged Perceptron Tagger downloaded successfully.")
        except Exception as e:
            print(f"Error downloading Averaged Perceptron Tagger: {e}")
# --------------------------

# --- Define Augmenters ---
print("Setting up Single Back-Translation Augmenter (En-Fr-En)...")
aug_single_bt = naw.BackTranslationAug(  # <--- Use naw.BackTranslationAug directly
    from_model_name='Helsinki-NLP/opus-mt-en-fr',
    to_model_name='Helsinki-NLP/opus-mt-fr-en',
    batch_size=256, # Internal nlpaug batch size (can be adjusted)
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=0
)
# Note: renamed variable to aug_single_bt for clarity
print("Single Back-Translation Augmenter ready.")

# 1. Back-Translation Augmenter (will run with num_proc=1)
#    Using chained BT as an example
# aug_chained_bt = naf.Sequential([
#     naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-es', to_model_name='Helsinki-NLP/opus-mt-es-en', batch_size=256, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=0),
#     naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-fr', to_model_name='Helsinki-NLP/opus-mt-fr-en', batch_size=256, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=0)
# ])

# 2. Other Noise Augmenters (will run with num_proc > 1)
aug_agg_char = naf.Sequential([
    nac.RandomCharAug(action="swap", aug_char_p=0.3, aug_word_p=0.3),
    nac.RandomCharAug(action="delete", aug_char_p=0.3, aug_word_p=0.3)
])
aug_agg_word = naf.Sequential([
    naw.RandomWordAug(action="swap", aug_p=0.3),
    naw.RandomWordAug(action="delete", aug_p=0.3)
])
aug_gibberish = nac.RandomCharAug(action="substitute", aug_char_p=0.6, aug_word_p=0.8)

# --- Define Augmentation Functions ---

# Function specifically for SINGLE back-translation - Batch Optimized
def augment_bt_only_batched(batch):
    """
    Applies the SINGLE back-translation augmenter (aug_single_bt)
    to a batch of texts provided by datasets.map.
    """
    original_texts = batch['text']

    if not original_texts:
        return {'augmented_text': []}

    try:
        # Directly augment the entire list (batch) of texts.
        # Use the NEW single BT augmenter variable name 'aug_single_bt'
        augmented_texts_batch = aug_single_bt.augment(original_texts)

        # Sanity check (still good practice)
        if len(augmented_texts_batch) != len(original_texts):
             print(f"Warning: Mismatch in SINGLE BT augmentation output length. Input: {len(original_texts)}, Output: {len(augmented_texts_batch)}. Returning originals.")
             return {'augmented_text': original_texts}

    except Exception as e:
        print(f"Single BT Batch Augmentation failed: {e}. Returning original batch.")
        augmented_texts_batch = original_texts

    return {'augmented_text': augmented_texts_batch}

def augment_other_probabilistic(batch):
    original_texts = batch['text']
    augmented_texts_batch = []
    for text in original_texts:
        if not text or not text.strip():
            augmented_texts_batch.append(text)
            continue

        # Choose ONE method from the OTHER augmenters
        chosen_aug = random.choices(other_aug_list, weights=other_probs_list, k=1)[0]

        if chosen_aug is None:
            augmented_texts_batch.append(text)
        else:
            try:
                augmented_text = chosen_aug.augment(text)
                augmented_texts_batch.append(augmented_text)
            except Exception as e:
                # print(f"Other Augmentation failed for text: {text[:50]}... Error: {e}") # Optional debugging
                augmented_texts_batch.append(text) # Keep original on failure
    return {'augmented_text': augmented_texts_batch}

# --- Add Labels Back ---
# Helper function to add labels back after augmentation map
def add_labels_back(augmented_subset: Dataset, original_labels: list) -> Dataset:
    """Adds the 'label' column back to an augmented dataset."""
    if len(augmented_subset) != len(original_labels):
         raise ValueError("Augmented subset size doesn't match original labels size!")

    def add_label_map(example, idx):
        example['label'] = original_labels[idx]
        return example

    return augmented_subset.map(add_label_map, with_indices=True)

# --- Main Script ---
def main():
    download_nltk_data()

    # Configuration
    TARGET_BT_FRACTION = 0.30 # Target fraction of data to augment with BT
    NUM_PROC_OTHER = 1       # Num_proc for non-BT augmentations
    SEED = 42
    random.seed(SEED) # Seed for splitting data

    # 1) Load Data
    dataset = load_dataset('ag_news', split='train')
    print("Loaded original 'ag_news' train split.")

    # 2) Split into Train/Eval
    # split_datasets = dataset.train_test_split(test_size=20000, seed=SEED)
    # original_train_dataset = split_datasets['train']
    original_train_dataset = dataset
    print(f"Original Train dataset size: {len(original_train_dataset)}")

    # 3) Split data for BT vs Other augmentations
    num_total = len(original_train_dataset)
    num_bt = math.ceil(num_total * TARGET_BT_FRACTION) # Use ceiling to ensure we cover the fraction
    num_other = num_total - num_bt

    # Shuffle indices to randomly select which samples get BT
    all_indices = list(range(num_total))
    random.shuffle(all_indices)
    bt_indices = all_indices[:num_bt]
    other_indices = all_indices[num_bt:]

    dataset_for_bt = original_train_dataset.select(bt_indices)
    dataset_for_other = original_train_dataset.select(other_indices)

    print(f"Selected {len(dataset_for_bt)} samples for Back-Translation (num_proc=1)")
    print(f"Selected {len(dataset_for_other)} samples for Other Augmentations (num_proc={NUM_PROC_OTHER})")

    # --- Stage 1: Back-Translation Augmentation ---
    print("\n--- Starting Stage 1: Back-Translation ---")
    # Keep original labels for this subset
    original_labels_bt = dataset_for_bt['label']
    # Apply BT augmentation
    augmented_bt_data = dataset_for_bt.map(
        augment_bt_only_batched,
        batched=True,
        batch_size=256,
        num_proc=1,
        remove_columns=dataset_for_bt.column_names
    )
    # Add labels back and cast schema
    augmented_bt_data = add_labels_back(augmented_bt_data, original_labels_bt)
    augmented_bt_data = augmented_bt_data.cast(original_train_dataset.features)
    print("--- Stage 1 Complete ---")

    # --- Stage 2: Other Augmentations ---
    print("\n--- Starting Stage 2: Other Augmentations ---")
     # Keep original labels for this subset
    original_labels_other = dataset_for_other['label']
    # Apply other augmentations probabilistically
    augmented_other_data = dataset_for_other.map(
        augment_other_probabilistic,
        batched=True,
        batch_size=256, # Can likely use larger batch size here
        num_proc=NUM_PROC_OTHER, # Use multiprocessing
        remove_columns=dataset_for_other.column_names
    )
    # Add labels back and cast schema
    augmented_other_data = add_labels_back(augmented_other_data, original_labels_other)
    augmented_other_data = augmented_other_data.cast(original_train_dataset.features)
    print("--- Stage 2 Complete ---")

    # --- Stage 3: Combine and Save ---
    print("\n--- Starting Stage 3: Combining Datasets ---")
    final_augmented_dataset = concatenate_datasets([augmented_bt_data, augmented_other_data])
    # Shuffle the final combined dataset
    final_augmented_dataset = final_augmented_dataset.shuffle(seed=SEED)

    print(f"\nFinal Augmented Train dataset size: {len(final_augmented_dataset)}")
    print("Final Augmented dataset features:", final_augmented_dataset.features)

    # Save the final combined dataset
    output_path = "augmented_harsh_two_stage_train.arrow" # New name
    final_augmented_dataset.save_to_disk(output_path)
    print(f"\nTwo-stage harsh augmentation complete. Dataset saved to '{output_path}'.")

if __name__ == "__main__":
    # try:
    #     mp.set_start_method("spawn", force=True)
    #     print("Set multiprocessing start method to 'spawn'.")
    # except RuntimeError:
    #     print("Could not set start method to 'spawn'.")
    main()