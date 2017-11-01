import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from util import generate_submission_ensemble

models = [
    ('inc_agg_aug_full_cp11', 1, 0),
    ('inc_agg_aug_full_cp11', 1, 1),
    ('inc_agg_aug_full_cp12', 1, 0),
    ('inc_agg_aug_full_cp12', 1, 1),
    ('inc_agg_aug_full_cp13', 1, 0),
    ('inc_agg_aug_full_cp13', 1, 1),
    ('inc_agg_aug_full_cp14', 1, 0),
    ('inc_agg_aug_full_cp14', 1, 1),
    ('inc_agg_aug_full_cp15', 1, 0),
    ('inc_agg_aug_full_cp15', 1, 1),
    ('inc_agg_aug_full_cp16', 1, 0),
    ('inc_agg_aug_full_cp16', 1, 1),
    ('inc_agg_aug_full_cp17', 1, 0),
    ('inc_agg_aug_full_cp17', 1, 1),
    ('inc_agg_aug_full_cp17', 1, 2),
    ('inc_agg_aug_full_cp18', 1, 0),
    ('inc_agg_aug_full_cp18', 1, 1),
    ('inc_agg_aug_full_cp18', 1, 2),
    ('inc_agg_aug_full_cp19', 1, 0),
    ('inc_agg_aug_full_cp19', 1, 1),
    ('inc_agg_aug_full_cp19', 1, 2),
    ('resnet50_augmentation_experiment_ths8', 1, 0),
    ('resnet50_augmentation_experiment_ths8', 1, 1),
    ('resnet50_augmentation_experiment_ths9', 1, 0),
    ('resnet50_augmentation_experiment_ths9', 1, 1),
    ('inc_agg_aug_full_gpu1_cp4', 1, 0),
    ('inc_agg_aug_full_gpu1_cp4', 1, 1),
    ('inc_agg_aug_full_gpu1_cp5', 1, 0),
    ('inc_agg_aug_full_gpu1_cp5', 1, 1),
]
generate_submission_ensemble('andrei', models)