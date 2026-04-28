"""
Step 1: Dataset Preparation
Organizes images into train/test folders and reports class balance.
Run this ONCE before training.
"""
import os
import shutil
from sklearn.model_selection import train_test_split


def organize_dataset(healthy_dir='Healthy', nutrient_dir='Nutrient', output_dir='dataset'):
    """
    Organizes raw images into train/test folder structure:
    dataset/
    ├── train/
    │   ├── Healthy/
    │   └── Nutrient/
    └── test/
        ├── Healthy/
        └── Nutrient/
    """
    for split in ['train', 'test']:
        for cls in ['Healthy', 'Nutrient']:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls, src_dir in [('Healthy', healthy_dir), ('Nutrient', nutrient_dir)]:
        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 80/20 train/test split
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

        for img in train_imgs:
            shutil.copy2(os.path.join(src_dir, img),
                         os.path.join(output_dir, 'train', cls, img))

        for img in test_imgs:
            shutil.copy2(os.path.join(src_dir, img),
                         os.path.join(output_dir, 'test', cls, img))

        print(f"  {cls}: {len(train_imgs)} train, {len(test_imgs)} test")


def check_class_balance(dataset_dir):
    """Check and report class distribution."""
    for cls in sorted(os.listdir(dataset_dir)):
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {cls}: {count} images")


if __name__ == '__main__':
    print("=" * 50)
    print("Step 1: Dataset Preparation")
    print("=" * 50)

    organize_dataset()

    print("\nClass distribution (train):")
    check_class_balance('dataset/train')

    print("\nClass distribution (test):")
    check_class_balance('dataset/test')

    print("\n✓ Dataset preparation complete!")
