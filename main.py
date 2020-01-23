from src.utils import plot_example
from src.food_dataset import FoodVisorDataset


dataset = FoodVisorDataset(
    json_annotations="configs/img_annotations.json",
    csv_mapping="configs/label_mapping.csv",
    imgs_folder="data/",
    regex_aliment="[Tt]omate(s)?",
)

plot_example(dataset, idx=1270)

