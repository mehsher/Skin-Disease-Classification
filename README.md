# Skin Disease Classification

Image classifier that predicts one of 8 skin conditions (bacterial, fungal, viral, and parasitic) from a photo, using transfer learning on MobileNetV2. Built as my final-year B.Tech major project.

## Classes

| Category | Conditions |
|---|---|
| Bacterial | Cellulitis, Impetigo |
| Fungal | Athlete's Foot, Nail Fungus, Ringworm |
| Viral | Chickenpox, Shingles |
| Parasitic | Cutaneous Larva Migrans |

## Approach

- Base model: MobileNetV2 (`keras.applications`, ImageNet weights, frozen)
- Custom dense classification head trained on top of the frozen feature extractor
- Images resized to 224x224 and scaled to [0, 1] before training
- Dataset: [Skin Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset) (Kaggle), pre-split into train/test folders by class

## Results

97% accuracy on the held-out test set (234 images), with 99.9% training accuracy — the small gap suggests the model is generalizing rather than memorizing.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Cellulitis | 0.97 | 1.00 | 0.99 |
| Impetigo | 0.95 | 1.00 | 0.98 |
| Athlete's Foot | 0.94 | 0.97 | 0.95 |
| Nail Fungus | 0.97 | 1.00 | 0.99 |
| Ringworm | 0.95 | 0.91 | 0.93 |
| Cutaneous Larva Migrans | 1.00 | 0.88 | 0.94 |
| Chickenpox | 1.00 | 0.97 | 0.99 |
| Shingles | 0.94 | 0.97 | 0.96 |

Ringworm and larva migrans have the lowest recall. Ringworm and athlete's foot are both fungal infections with visually similar red, scaly patches, so some of that confusion tracks with the clinical similarity between the two conditions, not just model error.

## A note on generalization (read this before trusting the 97% too much)

The 97% figure is accuracy on test images drawn from the *same* dataset as training. When I ran the model on unrelated images (photos pulled from Google, not from this dataset), performance dropped noticeably and was sometimes wrong with high confidence.

Auditing the dataset turned up why:

- Most classes contain images of wildly inconsistent original resolution (20-50+ distinct sizes per class before resizing), consistent with images scraped from many different sources
- Two classes (`cellulitis`, `nail-fungus`) are suspiciously *uniform* in original size across every image, and `nail-fungus` filenames follow a completely different naming convention than every other class — a strong signal that class was sourced separately from the rest and merged in under the same label
- Average pixel brightness also differs measurably between the dataset's own train and test splits

None of this is something code can fix. It means the model has some room to key off dataset/source artifacts rather than the disease's actual visual features, which is the most likely explanation for the gap between in-dataset accuracy and real-world performance. A more rigorous next step would be re-collecting or heavily auditing the source images per class, or validating against a second, independently-sourced dataset (e.g. ISIC) before trusting this for anything beyond a learning exercise.

I'm including this because I think knowing where a model's evaluation number stops meaning what it looks like it means is a more useful skill than reporting a clean number and stopping there.

## Demo

A small Gradio app at the end of the notebook takes an uploaded image and returns the top-3 predicted conditions with confidence scores.

## Running it

1. Download the dataset from Kaggle (link above)
2. Update `dataset_url` in the notebook to point at your local or Kaggle path
3. Run top to bottom — a GPU is recommended for the training cell (~15 epochs, a few minutes on a T4)

```bash
pip install -r requirements.txt
```

## Repo structure

```
notebooks/
  skin_disease_classification.ipynb   # data loading -> training -> evaluation -> demo
requirements.txt
```

## Known limitations

- Small dataset (roughly 100-150 images per class) with inconsistent per-class sourcing, detailed above
- Not validated against an independent, held-out dataset from a different source
- This is a learning project, not a diagnostic tool, and shouldn't be used to make real medical decisions

## Author

Syed Mehsher
