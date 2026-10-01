"""# Imports"""

import argparse
import glob
import numpy as np
import os
import pandas as pd
from scipy import stats

from skimage import measure

import nibabel as nib

import pymeshlab as ml

from ExperimentBuilder import ExperimentBuilder

parser = argparse.ArgumentParser(description="Create meshs per Fold")
parser.add_argument('--pred_path', type=str, required=True, help='pred_path')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
parser.add_argument('--labels', nargs='+', required=True, help='labels')
args = parser.parse_args()

"""# Experiment setup"""

approach = 'resized'

pred_path = args.pred_path
print(pred_path)
pred_id = os.path.basename(pred_path)
print(pred_id)

dataset_name = args.dataset
label = next((l for l in args.labels if l in pred_id))

if '_' in approach:
    approach_tag = approach.partition('_')[0]
else:
    approach_tag = approach

setup = {
    'experiment_id': pred_id,
    'accelerator': None,
    'devices': None,
    'dataset_name': dataset_name,
    'dataset_path': f'../data/datasets/{dataset_name}_{approach_tag}',
    'groud_truth_root': f'../data/datasets/{dataset_name}_{approach_tag}/ground_truths',
    'target_label': label,
    'datamodule': None,
    'transforms_set': None,
    'hu_range': None,
    'task': None,
    'architecture': None,
    'in_channels': None,
    'num_classes': None,
    'img_size': None,
    'loss': None,
    'seed': 42,
    'batch_size': None,
    'accumulate_grad_batches': None,
    'num_workers': None,
    'one_cycle_lr': None,
    'base_lr': None,
    'weight_decay': None,
    'max_lr': None,
    'pct_start': None,
    'div_factor': None,
    'final_div_factor': None,
    'max_epochs': None,
    'k-fold': 5,
    'save_top_k': None,
    'monitor_metric': None,
    'monitor_mode': None,
    'log_every_n_steps': None,
    'check_val_every_n_epoch': None,
    'num_imgs_to_plot': None
}


mesh_path = f'meshs/{setup["dataset_name"]}/{setup["target_label"]}/{setup["experiment_id"].replace("pred", "mesh")}'
os.makedirs(os.path.join(mesh_path), exist_ok=True)

"""# Experiment builder"""

experiment = ExperimentBuilder(setup, mesh_path, save_setup=False)

"""# Functions"""

"""## Get mesh from volume"""

def get_mesh_from_volume(volume_path, out_mesh_path, id_flag, target_ids, step_size=1):

    nii = nib.load(volume_path)
    data = nii.get_fdata()
    affine = nii.affine

    # print(f"Shape do volume salvo: {data.shape}")

    ids = 1 if id_flag == 'pred' else target_ids
    data = np.isin(data, ids).astype(np.uint8)

    if np.max(data) == 0:
        return False 

    verts, faces, normals, _ = measure.marching_cubes(
        data,
        level=0.5,
        step_size=step_size,
        allow_degenerate=False,
        gradient_direction="ascent"
    )

    verts = nib.affines.apply_affine(affine, verts)

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=verts, face_matrix=faces))
    ms.save_current_mesh(out_mesh_path)

    return True

"""## Process meshs"""

def process_prediction_mesh(input_mesh, output_final_path, debug_dir=None):
    
    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh)

    def save_debug(name):
        if debug_dir:
            ms.save_current_mesh(os.path.join(debug_dir, name))

    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    
    ms.meshing_remove_unreferenced_vertices()

    ms.compute_selection_point_cloud_outliers(knearest=50)
    ms.meshing_remove_selected_vertices()
    
    save_debug("01_pred_cleaned_marching_cubes.ply")

    ms.compute_normal_per_vertex()

    ms.compute_selection_by_condition_per_vertex(condselect='(nx==0) && (ny==0) && (nz==0)')
    ms.meshing_remove_selected_vertices()

    ms.generate_surface_reconstruction_screened_poisson(
        depth=10,
        scale=1.2,
        samplespernode=1.5,
        pointweight=2.0,
        preclean=True
    )
    save_debug("02_pred_poisson_reconstruction.ply")

    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    
    ms.meshing_remove_unreferenced_vertices()

    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=20)
    save_debug("03_pred_taubin_smooth.ply")

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=50000,
        preservenormal=True,
        preservetopology=True
    )

    ms.compute_normal_per_vertex()

    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=5)
    save_debug("04_pred_final_polished_mesh.ply")

    ms.save_current_mesh(output_final_path)
    # print(f"Prediction mesh saved: {os.path.basename(output_final_path)}")
    
    return output_final_path
    
def process_ground_truth_mesh(input_mesh, output_final_path, debug_dir=None):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh)

    def save_debug(name):
        if debug_dir:
            ms.save_current_mesh(os.path.join(debug_dir, name))

    ms.meshing_remove_unreferenced_vertices()

    ms.compute_normal_per_vertex()
    
    ms.compute_selection_by_condition_per_vertex(condselect='(nx==0) && (ny==0) && (nz==0)')
    ms.meshing_remove_selected_vertices()

    ms.generate_surface_reconstruction_screened_poisson(
        depth=10,
        scale=1.2,
        samplespernode=1.5,
        pointweight=2.0,
        preclean=True
    )
    save_debug("01_gt_poisson_reconstruction.ply")

    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()

    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=20)
    save_debug("02_gt_taubin_smooth.ply")

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=50000,
        preservenormal=True,
        preservetopology=True
    )

    ms.compute_normal_per_vertex()

    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=5)
    save_debug("03_gt_final_polished_mesh.ply")

    ms.save_current_mesh(output_final_path)
    # print(f"Ground truth mesh saved: {os.path.basename(output_final_path)}")
    
    return output_final_path
    
"""## Compute Hausdorff"""

def compute_all_surface_metrics(gt_mesh_path, pred_mesh_path, out_vis_path=None, max_color_dist=10.0):

    ms = ml.MeshSet()
    
    ms.load_new_mesh(pred_mesh_path) # Mesh ID 0 (Pred)
    ms.load_new_mesh(gt_mesh_path)   # Mesh ID 1 (GT)

    res_pred_gt = ms.apply_filter(
        "get_hausdorff_distance",
        sampledmesh=0,
        targetmesh=1
    )
    dist_pred = ms.mesh(0).vertex_scalar_array()

    res_gt_pred = ms.apply_filter(
        "get_hausdorff_distance",
        sampledmesh=1,
        targetmesh=0
    )
    dist_gt = ms.mesh(1).vertex_scalar_array()

    hd = max(res_pred_gt['max'], res_gt_pred['max'])
    
    all_distances = np.concatenate([dist_pred, dist_gt])
    
    hd95 = np.percentile(all_distances, 95)
    
    sd = np.mean(all_distances)

    if out_vis_path:
        ms.set_current_mesh(0) 
        ms.apply_filter(
            "compute_color_from_scalar_per_vertex",
            colormap='Turbo',
            minval=0.0,
            maxval=max_color_dist
        )
        ms.save_current_mesh(out_vis_path)
        
        # print(f"Distance heatmap saved: {os.path.basename(out_vis_path)}")

    return {
        "HD": float(hd),
        "HD95": float(hd95),
        "SD": float(sd)
    }
    
"""# Run"""

all_folds_summary = []  

for k in range(setup['k-fold']):
    
    base_out_fold = os.path.join(mesh_path, f'fold_{k}')
    
    out_pred_meshes = os.path.join(base_out_fold, "predicted_meshes")
    out_gt_meshes = os.path.join(base_out_fold, "ground_truth_meshes")
    out_vis_root = os.path.join(base_out_fold, "visualizations")
    
    os.makedirs(out_pred_meshes, exist_ok=True)
    os.makedirs(out_gt_meshes, exist_ok=True)
    os.makedirs(out_vis_root, exist_ok=True)
    
    print(f'\nFold {k}')
    
    results_list = [] 

    fold_pred_path = os.path.join(pred_path, f'fold_{k}')
    
    for filename in sorted(os.listdir(fold_pred_path)):
        
        if not filename.endswith(".nii.gz"):
            continue
    
        print(f"Processing: {filename}")
    
        clean_name = filename.replace("pred_", "").replace(".nii.gz", "")
        volume_id = clean_name
    
        pred_nii_path = os.path.join(fold_pred_path, filename)
        gt_nii_path = os.path.join(setup['groud_truth_root'], filename.replace("pred_", ""))
    
        if not os.path.exists(gt_nii_path):
            print(f"Ground truth not found: {clean_name}")
            continue
    
        final_pred_mesh_path = os.path.join(out_pred_meshes, f"{clean_name}.ply")
        final_gt_mesh_path = os.path.join(out_gt_meshes, f"{clean_name}_gt.ply")
    
        visualization_dir = os.path.join(out_vis_root, volume_id)
        os.makedirs(visualization_dir, exist_ok=True)
    
        raw_pred_ply = os.path.join(visualization_dir, "raw_pred.ply")
        raw_gt_ply = os.path.join(visualization_dir, "raw_gt.ply")
    
        target_ids = experiment._get_dataset_labels(setup['dataset_path'])[setup['target_label']]
    
        has_pred = get_mesh_from_volume(pred_nii_path, raw_pred_ply, 'pred', target_ids)
        has_gt = get_mesh_from_volume(gt_nii_path, raw_gt_ply, 'gt', target_ids)

        if not has_pred or not has_gt:
            print(f"Volume vazio (GT ou Pred) para: {clean_name}. Pulando criação de malha.")
            
            results_list.append({
                'volume_id': volume_id,
                'HD': np.nan,
                'HD95': np.nan,
                'SD': np.nan
            })
            
            continue
    
        process_prediction_mesh(raw_pred_ply, final_pred_mesh_path, debug_dir=visualization_dir)
        process_ground_truth_mesh(raw_gt_ply, final_gt_mesh_path, debug_dir=visualization_dir)
    
        distance_vis_path = os.path.join(visualization_dir, "distance_heatmap.ply")
        
        metrics = compute_all_surface_metrics(
            gt_mesh_path=final_gt_mesh_path, 
            pred_mesh_path=final_pred_mesh_path, 
            out_vis_path=distance_vis_path
        )
        
        results_list.append({
            'volume_id': volume_id,
            'HD': metrics['HD'],
            'HD95': metrics['HD95'],
            'SD': metrics['SD']
        })

    df_metrics = pd.DataFrame(results_list)
    cols = ['volume_id', 'HD', 'HD95', 'SD']
    df_metrics = df_metrics[cols]
      
    csv_path = os.path.join(base_out_fold, f"surface_metrics_instances_fold_{k}.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")

    fold_mean_metrics = df_metrics[['HD', 'HD95', 'SD']].mean().to_dict()
    fold_mean_metrics['Fold'] = k
    fold_mean_metrics['Experiment_ID'] = setup.get("experiment_id", "exp_default")
    
    all_folds_summary.append(fold_mean_metrics)

df_results = pd.DataFrame(all_folds_summary)

df_results.replace([np.inf, -np.inf], np.nan, inplace=True)

"""# Stats"""

if 'HD95' in df_results.columns and 'HD' in df_results.columns:
    df_results['HD95_HD_Ratio'] = df_results['HD95'] / df_results['HD']

print("\nT-Student")

metric_cols = ['HD', 'HD95', 'HD95_HD_Ratio', 'SD']

n_folds = len(df_results)
degrees_of_freedom = n_folds - 1
confidence_level = 0.95
t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

summary_stats = []

for col in metric_cols:

    values = pd.to_numeric(df_results[col]) if df_results[col].dtype == object else df_results[col]
    
    mean = values.mean()
    std_dev = values.std(ddof=1) 
    std_error = std_dev / np.sqrt(n_folds)
    margin_of_error = t_critical * std_error

    summary_stats.append({
        'Metric': col,
        'Mean': mean,
        'Std': std_dev,
        'Error': std_error,
        'CI_95%_Lower': mean - margin_of_error,
        'CI_95%_Upper': mean + margin_of_error,
        'Margin_Error': margin_of_error 
    })

"""# Format and save"""

df_results_fmt = df_results.copy()

cols = ['Fold', 'Experiment_ID'] + [c for c in df_results.columns if c not in ['Fold', 'Experiment_ID', 'HD95_HD_Ratio', 'SD']]

if 'HD95_HD_Ratio' in df_results_fmt.columns:
    cols.append('HD95_HD_Ratio')
if 'SD' in df_results_fmt.columns:
    cols.append('SD')

df_results_fmt = df_results_fmt[cols]

for col in metric_cols:
    if col in df_results_fmt.columns:
        df_results_fmt[col] = df_results_fmt[col].map(lambda x: f"{x:.2f}".replace('.', ','))

csv_filename = os.path.join(mesh_path, 'surface_metrics_folds_summary.csv')
df_results_fmt.to_csv(csv_filename, index=False)
print(f"Summary saved: {csv_filename}")

df_summary = pd.DataFrame(summary_stats)

def final_formatting(row):

    fmt_main = lambda x: f"{x:.2f}".replace('.', ',')
    
    return pd.Series({
        'Metric': row['Metric'],
        'Mean': fmt_main(row['Mean']),
        'Std': fmt_main(row['Std']),
        'Error': fmt_main(row['Error']),
        'CI_95%_Lower': fmt_main(row['CI_95%_Lower']),
        'CI_95%_Upper': fmt_main(row['CI_95%_Upper']),
        'Report': f"{fmt_main(row['Mean'])} ± {fmt_main(row['Margin_Error'])}"
    })

df_summary_final = df_summary.apply(final_formatting, axis=1)

summary_filename = os.path.join(mesh_path, 'surface_metrics_confidence_intervals.csv')
df_summary_final.to_csv(summary_filename, index=False)
print(f"Confidence intervals saved: {summary_filename}")
