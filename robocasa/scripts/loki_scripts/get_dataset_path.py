from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS
from robocasa.utils.dataset_registry import get_ds_path

# iterate through all atomic and composite tasks
for task in list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS):
    print(f"Datasets for {task}")
    human_path, ds_meta = get_ds_path(task=task, ds_type="human_im", return_info=True)  # human dataset path
    horizon = ds_meta["horizon"]                                      # get suggested for dataset
    mg_path, ds_meta = get_ds_path(task=task, ds_type="mg_im", return_info=True)        # MimicGen dataset path
    print(f"Human ds: {human_path}")
    print(f"MimicGen ds: {mg_path}")
    print(f"Dataset horizon:", horizon)
    print()


