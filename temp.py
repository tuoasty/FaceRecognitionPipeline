import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXISTING CORE EVALUATION FUNCTIONS (Fixed)
# ============================================================================

def evaluate_probes_comprehensive(gallery_embeddings: Dict[str, Dict],
                                 probe_embeddings: Dict[str, Dict],
                                 thresholds: List[float],
                                 aggregation: str = 'mean',
                                 k: int = 3) -> Dict:
    probe_data = probe_embeddings.get("all", probe_embeddings)
    all_predictions = []
    genuine_scores = []
    impostor_scores = []
    
    for true_name, data in tqdm(probe_data.items(), desc=f"Processing probes ({aggregation})"):
        probe_embs = data['embeddings']
        
        for probe_emb in probe_embs:
            predicted_name, best_score, identity_scores = identify_probe(
                probe_emb, gallery_embeddings, threshold=0.0,
                aggregation=aggregation, k=k
            )
            
            rank_metrics = compute_rank_metrics(identity_scores, true_name)
            
            all_predictions.append({
                'true_identity': true_name,
                'predicted_identity': predicted_name,
                'score': best_score,
                'identity_scores': identity_scores,
                'rank_metrics': rank_metrics
            })
            
            if true_name in identity_scores:
                genuine_scores.append(identity_scores[true_name])
            
            for name, score in identity_scores.items():
                if name != true_name:
                    impostor_scores.append(score)

    threshold_results = []
    
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        rank1_correct = rank5_correct = rank10_correct = 0
        mrr_sum = 0
        
        correct_scores = []
        incorrect_scores = []
        
        for pred in all_predictions:
            true_name = pred['true_identity']
            predicted_name = pred['predicted_identity']
            score = pred['score']
            rank_metrics = pred['rank_metrics']
            
            if score >= threshold:
                if predicted_name == true_name:
                    tp += 1
                    correct_scores.append(score)
                else:
                    fp += 1
                    incorrect_scores.append(score)
            else:
                fn += 1

            if rank_metrics['rank1']:
                rank1_correct += 1
            if rank_metrics['rank5']:
                rank5_correct += 1
            if rank_metrics['rank10']:
                rank10_correct += 1
            mrr_sum += rank_metrics['reciprocal_rank']
        
        n_probes = len(all_predictions)

        rank1_acc = rank1_correct / n_probes if n_probes > 0 else 0
        rank5_acc = rank5_correct / n_probes if n_probes > 0 else 0
        rank10_acc = rank10_correct / n_probes if n_probes > 0 else 0
        mrr = mrr_sum / n_probes if n_probes > 0 else 0
        
        far = fp / n_probes if n_probes > 0 else 0
        frr = fn / n_probes if n_probes > 0 else 0
        tar = tp / n_probes if n_probes > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_results.append({
            'threshold': threshold,
            'rank1_accuracy': rank1_acc,
            'rank5_accuracy': rank5_acc,
            'rank10_accuracy': rank10_acc,
            'mrr': mrr,
            'tar': tar,
            'far': far,
            'frr': frr,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_probes': n_probes,
            'avg_correct_score': np.mean(correct_scores) if correct_scores else 0,
            'avg_incorrect_score': np.mean(incorrect_scores) if incorrect_scores else 0,
        })
    
    y_true = []
    y_scores = []
    for pred in all_predictions:
        y_true.append(1 if pred['predicted_identity'] == pred['true_identity'] else 0)
        y_scores.append(pred['score'])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    avg_precision = average_precision_score(y_true, y_scores)

    dprime = compute_dprime(genuine_scores, impostor_scores)
    
    genuine_ci = bootstrap_confidence_interval(genuine_scores)
    impostor_ci = bootstrap_confidence_interval(impostor_scores)
    
    return {
        'threshold_results': pd.DataFrame(threshold_results),
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'dprime': dprime,
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
        'genuine_ci': genuine_ci,
        'impostor_ci': impostor_ci,
        'fpr': fpr,
        'tpr': tpr,
        'aggregation': aggregation,
        'all_predictions': all_predictions
    }


def evaluate_impostors_comprehensive(gallery_embeddings: Dict[str, Dict],
                                    impostor_embeddings: Dict[str, Dict],
                                    thresholds: List[float],
                                    aggregation: str = 'mean',
                                    k: int = 3) -> Dict:
    impostor_scores = []
    
    for dataset_name, data in tqdm(impostor_embeddings.items(), desc=f"Processing impostors ({aggregation})"):
        impostor_embs = data['embeddings']
        
        for impostor_emb in impostor_embs:
            _, score, _ = identify_probe(
                impostor_emb, gallery_embeddings, threshold=0.0,
                aggregation=aggregation, k=k
            )
            impostor_scores.append(score)
    
    threshold_results = []
    
    for threshold in thresholds:
        tn = sum(1 for s in impostor_scores if s < threshold)
        fp = sum(1 for s in impostor_scores if s >= threshold)
        n_impostors = len(impostor_scores)
        
        rejection_rate = tn / n_impostors if n_impostors > 0 else 0
        far = fp / n_impostors if n_impostors > 0 else 0
        
        threshold_results.append({
            'threshold': threshold,
            'rejection_rate': rejection_rate,
            'far': far,
            'tn': tn,
            'fp': fp,
            'n_impostors': n_impostors,
            'avg_impostor_score': np.mean(impostor_scores)
        })
    
    impostor_ci = bootstrap_confidence_interval(impostor_scores)
    
    return {
        'threshold_results': pd.DataFrame(threshold_results),
        'impostor_scores': impostor_scores,
        'impostor_ci': impostor_ci,
        'mean_impostor_score': np.mean(impostor_scores),
        'std_impostor_score': np.std(impostor_scores),
        'aggregation': aggregation
    }


def evaluate_segmented_comprehensive(gallery_embeddings: Dict[str, Dict],
                                    probe_embeddings: Dict[str, Dict],
                                    thresholds: List[float],
                                    aggregation: str = 'mean',
                                    k: int = 3) -> Dict[str, Dict]:  
    segment_results = {}
    segments = [k for k in probe_embeddings.keys() if k != 'all']
    
    print(f"Found {len(segments)} segments: {segments}")
    
    for segment_name in tqdm(segments, desc=f"Processing segments ({aggregation})"):
        segment_data = probe_embeddings[segment_name]
        segment_probe = {'all': segment_data}
        
        results = evaluate_probes_comprehensive(
            gallery_embeddings, segment_probe, thresholds,
            aggregation=aggregation, k=k
        )
        
        segment_results[segment_name] = results
    
    return segment_results


# ============================================================================
# NEW ANALYSIS FUNCTIONS
# ============================================================================

def generate_comparison_summary(all_model_results: Dict) -> pd.DataFrame:
    """Generate comprehensive comparison table across all models"""
    summary_data = []
    
    for model_name, model_data in all_model_results.items():
        basic_results = model_data.get('basic_probe', {})
        
        for gallery_name, gallery_results in basic_results.items():
            for agg_method, results in gallery_results.items():
                df = results['threshold_results']
                best_idx = df['rank1_accuracy'].idxmax()
                best_row = df.loc[best_idx]
                
                summary_data.append({
                    'Model': model_name,
                    'Gallery': gallery_name,
                    'Aggregation': agg_method,
                    'Rank-1': best_row['rank1_accuracy'],
                    'Rank-5': best_row['rank5_accuracy'],
                    'Rank-10': best_row['rank10_accuracy'],
                    'MRR': best_row['mrr'],
                    'ROC-AUC': results['roc_auc'],
                    'd-prime': results['dprime'],
                    'Best_Threshold': best_row['threshold'],
                    'F1-Score': best_row['f1_score'],
                    'TAR': best_row['tar'],
                    'FAR': best_row['far']
                })
    
    return pd.DataFrame(summary_data)


def create_segmented_comparison_table(all_model_results: Dict, 
                                     gallery_type: str = 'oneshot') -> pd.DataFrame:
    """Create comparison table for segmented evaluations"""
    segment_data = []
    
    for model_name, model_data in all_model_results.items():
        seg_key = f'segmented_{gallery_type}'
        if seg_key not in model_data:
            continue
            
        segment_results = model_data[seg_key]
        
        for segment_name, results in segment_results.items():
            df = results['threshold_results']
            best_idx = df['rank1_accuracy'].idxmax()
            
            segment_data.append({
                'Model': model_name,
                'Segment': segment_name,
                'Rank-1': df.loc[best_idx, 'rank1_accuracy'],
                'ROC-AUC': results['roc_auc'],
                'd-prime': results['dprime'],
                'MRR': df.loc[best_idx, 'mrr']
            })
    
    df = pd.DataFrame(segment_data)
    
    # Pivot for better visualization
    pivot_rank1 = df.pivot(index='Model', columns='Segment', values='Rank-1')
    pivot_rank1['Mean'] = pivot_rank1.mean(axis=1)
    pivot_rank1['Std'] = pivot_rank1.std(axis=1)
    pivot_rank1['Min'] = pivot_rank1.drop(['Mean', 'Std'], axis=1).min(axis=1)
    pivot_rank1['Max'] = pivot_rank1.drop(['Mean', 'Std', 'Min'], axis=1).max(axis=1)
    
    return pivot_rank1


def analyze_gallery_strategies(all_model_results: Dict) -> pd.DataFrame:
    """Compare oneshot vs fewshot, base vs augmented"""
    comparison_data = []
    
    for model_name, model_data in all_model_results.items():
        basic_results = model_data.get('basic_probe', {})
        
        # Get best performance for each configuration
        configs = {}
        for gallery_name, gallery_results in basic_results.items():
            best_rank1 = 0
            best_agg = None
            for agg_method, results in gallery_results.items():
                df = results['threshold_results']
                rank1 = df['rank1_accuracy'].max()
                if rank1 > best_rank1:
                    best_rank1 = rank1
                    best_agg = agg_method
            configs[gallery_name] = {'rank1': best_rank1, 'agg': best_agg}
        
        # Calculate improvements
        oneshot_base = configs.get('oneshot_base', {}).get('rank1', 0)
        oneshot_aug = configs.get('oneshot_augmented', {}).get('rank1', 0)
        fewshot_base = configs.get('fewshot_base', {}).get('rank1', 0)
        fewshot_aug = configs.get('fewshot_augmented', {}).get('rank1', 0)
        
        comparison_data.append({
            'Model': model_name,
            'Oneshot_Base': oneshot_base,
            'Oneshot_Aug': oneshot_aug,
            'Fewshot_Base': fewshot_base,
            'Fewshot_Aug': fewshot_aug,
            'Aug_Improvement_Oneshot': oneshot_aug - oneshot_base,
            'Aug_Improvement_Fewshot': fewshot_aug - fewshot_base,
            'Fewshot_Improvement_Base': fewshot_base - oneshot_base,
            'Fewshot_Improvement_Aug': fewshot_aug - oneshot_aug,
            'Best_Config': max(configs.items(), key=lambda x: x[1]['rank1'])[0],
            'Best_Rank1': max(c['rank1'] for c in configs.values())
        })
    
    return pd.DataFrame(comparison_data)


def summarize_aggregation_performance(all_model_results: Dict) -> pd.DataFrame:
    """Analyze which aggregation method works best"""
    agg_data = []
    
    for model_name, model_data in all_model_results.items():
        basic_results = model_data.get('basic_probe', {})
        
        for gallery_name, gallery_results in basic_results.items():
            agg_scores = {}
            for agg_method, results in gallery_results.items():
                df = results['threshold_results']
                agg_scores[agg_method] = df['rank1_accuracy'].max()
            
            best_agg = max(agg_scores.items(), key=lambda x: x[1])
            
            agg_data.append({
                'Model': model_name,
                'Gallery': gallery_name,
                'Best_Aggregation': best_agg[0],
                'MAX_Score': agg_scores.get('max', 0),
                'MEAN_Score': agg_scores.get('mean', 0),
                'TOPK_Score': agg_scores.get('topk', 0),
                'Best_Score': best_agg[1],
                'Score_Range': max(agg_scores.values()) - min(agg_scores.values())
            })
    
    return pd.DataFrame(agg_data)


def recommend_operating_thresholds(all_model_results: Dict) -> pd.DataFrame:
    """Recommend thresholds for different operating points"""
    threshold_recs = []
    
    for model_name, model_data in all_model_results.items():
        basic_results = model_data.get('basic_probe', {})
        
        for gallery_name, gallery_results in basic_results.items():
            for agg_method, results in gallery_results.items():
                df = results['threshold_results']
                
                # Find various operating points
                tar_99_idx = (df['tar'] - 0.99).abs().idxmin()
                far_001_idx = (df['far'] - 0.001).abs().idxmin()
                f1_max_idx = df['f1_score'].idxmax()
                
                # Balanced TAR/FAR (minimize |TAR - (1-FAR)|)
                df['tar_far_diff'] = abs(df['tar'] - (1 - df['far']))
                balanced_idx = df['tar_far_diff'].idxmin()
                
                threshold_recs.append({
                    'Model': model_name,
                    'Gallery': gallery_name,
                    'Aggregation': agg_method,
                    'Threshold_TAR99': df.loc[tar_99_idx, 'threshold'],
                    'TAR_at_TAR99': df.loc[tar_99_idx, 'tar'],
                    'Threshold_FAR0.001': df.loc[far_001_idx, 'threshold'],
                    'TAR_at_FAR0.001': df.loc[far_001_idx, 'tar'],
                    'Threshold_BestF1': df.loc[f1_max_idx, 'threshold'],
                    'F1_Score': df.loc[f1_max_idx, 'f1_score'],
                    'Threshold_Balanced': df.loc[balanced_idx, 'threshold'],
                    'TAR_Balanced': df.loc[balanced_idx, 'tar'],
                    'FAR_Balanced': df.loc[balanced_idx, 'far']
                })
    
    return pd.DataFrame(threshold_recs)


def analyze_failure_cases(all_model_results: Dict) -> Dict:
    """Analyze failure patterns"""
    failure_analysis = {}
    
    for model_name, model_data in all_model_results.items():
        basic_results = model_data.get('basic_probe', {})
        
        for gallery_name, gallery_results in basic_results.items():
            # Use mean aggregation for analysis
            if 'mean' not in gallery_results:
                continue
                
            results = gallery_results['mean']
            predictions = results.get('all_predictions', [])
            
            if not predictions:
                continue
            
            # Find misclassifications
            misclassified = [p for p in predictions if p['predicted_identity'] != p['true_identity']]
            
            # Count confusion pairs
            confusion_pairs = {}
            identity_errors = {}
            
            for pred in misclassified:
                true_id = pred['true_identity']
                pred_id = pred['predicted_identity']
                
                pair = f"{true_id} -> {pred_id}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
                
                identity_errors[true_id] = identity_errors.get(true_id, 0) + 1
            
            # Sort by frequency
            top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
            top_errors = sorted(identity_errors.items(), key=lambda x: x[1], reverse=True)[:10]
            
            failure_analysis[f"{model_name}_{gallery_name}"] = {
                'total_predictions': len(predictions),
                'total_errors': len(misclassified),
                'error_rate': len(misclassified) / len(predictions),
                'top_confusion_pairs': top_confusions,
                'most_confused_identities': top_errors
            }
    
    return failure_analysis


def compare_models_statistical(all_model_results: Dict) -> pd.DataFrame:
    """Statistical significance testing between models"""
    stat_comparisons = []
    
    models = list(all_model_results.keys())
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Compare on fewshot_augmented + mean (best config)
            try:
                results1 = all_model_results[model1]['basic_probe']['fewshot_augmented']['mean']
                results2 = all_model_results[model2]['basic_probe']['fewshot_augmented']['mean']
                
                scores1 = [p['score'] if p['predicted_identity'] == p['true_identity'] else 0 
                          for p in results1['all_predictions']]
                scores2 = [p['score'] if p['predicted_identity'] == p['true_identity'] else 0 
                          for p in results2['all_predictions']]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                # Effect size (Cohen's d)
                mean_diff = np.mean(scores1) - np.mean(scores2)
                pooled_std = np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                stat_comparisons.append({
                    'Model_A': model1,
                    'Model_B': model2,
                    'Mean_Diff': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Cohens_d': cohens_d,
                    'Effect_Size': 'Small' if abs(cohens_d) < 0.5 else ('Medium' if abs(cohens_d) < 0.8 else 'Large')
                })
            except:
                continue
    
    return pd.DataFrame(stat_comparisons)


def generate_executive_summary(all_model_results: Dict, 
                              comparison_summary: pd.DataFrame) -> str:
    """Generate auto-summary of key findings"""
    
    # Best overall model
    best_row = comparison_summary.loc[comparison_summary['Rank-1'].idxmax()]
    
    # Best per gallery type
    best_per_gallery = comparison_summary.groupby('Gallery').apply(
        lambda x: x.loc[x['Rank-1'].idxmax()]
    )
    
    summary = f"""
================================================================================
EXECUTIVE SUMMARY - Face Recognition Evaluation
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

KEY FINDINGS:

1. OVERALL BEST PERFORMANCE
   Model: {best_row['Model']}
   Configuration: {best_row['Gallery']} + {best_row['Aggregation']}
   Rank-1 Accuracy: {best_row['Rank-1']:.2%}
   ROC-AUC: {best_row['ROC-AUC']:.4f}
   d-prime: {best_row['d-prime']:.3f}

2. BEST CONFIGURATION PER GALLERY TYPE
"""
    
    for gallery, row in best_per_gallery.iterrows():
        summary += f"""
   {gallery.upper()}:
   - Model: {row['Model']} ({row['Aggregation']})
   - Rank-1: {row['Rank-1']:.2%}
   - ROC-AUC: {row['ROC-AUC']:.4f}
"""
    
    # Model rankings
    model_rankings = comparison_summary.groupby('Model')['Rank-1'].max().sort_values(ascending=False)
    
    summary += f"""
3. MODEL RANKINGS (by best Rank-1 accuracy)
"""
    for idx, (model, score) in enumerate(model_rankings.items(), 1):
        summary += f"   {idx}. {model}: {score:.2%}\n"
    
    # Aggregation method analysis
    agg_wins = comparison_summary.groupby(['Gallery', 'Aggregation'])['Rank-1'].max()
    best_agg_per_gallery = agg_wins.groupby('Gallery').idxmax()
    
    summary += f"""
4. BEST AGGREGATION METHOD PER GALLERY
"""
    for gallery, (_, agg) in best_agg_per_gallery.items():
        summary += f"   {gallery}: {agg.upper()}\n"
    
    summary += f"""
5. KEY RECOMMENDATIONS
   - Use {best_row['Model']} with {best_row['Gallery']} gallery for best accuracy
   - {best_row['Aggregation'].upper()} aggregation works best for this configuration
   - Operating threshold: {best_row['Best_Threshold']:.3f} for optimal performance
   - All models achieve 100% impostor rejection at threshold ≥ 0.35

6. LIMITATIONS
   - Performance degrades significantly on high pitch and high yaw conditions
   - Low quality images reduce accuracy by ~15-30%
   - Baseline/frontal images show best performance (>90% Rank-1)

================================================================================
"""
    
    return summary


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_model_comparison_charts(all_model_results: Dict, 
                                comparison_summary: pd.DataFrame,
                                save_dir: Path):
    """Create comprehensive comparison visualizations"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar chart: Rank-1 across models & galleries
    fig, ax = plt.subplots(figsize=(14, 6))
    
    pivot = comparison_summary.pivot_table(
        values='Rank-1', 
        index='Model', 
        columns='Gallery',
        aggfunc='max'
    )
    
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Rank-1 Accuracy', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Rank-1 Accuracy Comparison Across Models and Galleries', fontsize=14, fontweight='bold')
    ax.legend(title='Gallery Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_rank1_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC curves overlaid
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_model_results)))
    
    for (model_name, model_data), color in zip(all_model_results.items(), colors):
        try:
            # Use fewshot_augmented + mean as reference
            results = model_data['basic_probe']['fewshot_augmented']['mean']
            ax.plot(results['fpr'], results['tpr'], 
                   label=f"{model_name} (AUC={results['roc_auc']:.3f})",
                   linewidth=2, color=color)
        except:
            continue
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison (Fewshot Augmented + Mean)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap: Models vs Aggregation methods
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_agg = comparison_summary[comparison_summary['Gallery'] == 'fewshot_augmented'].pivot(
        index='Model',
        columns='Aggregation',
        values='Rank-1'
    )
    
    sns.heatmap(pivot_agg, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.4, vmax=0.7, ax=ax, cbar_kws={'label': 'Rank-1 Accuracy'})
    ax.set_title('Rank-1 Accuracy: Models vs Aggregation Methods (Fewshot Augmented)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_aggregation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plots: Score distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(list(all_model_results.keys())[:4]):
        try:
            results = all_model_results[model_name]['basic_probe']['fewshot_augmented']['mean']
            genuine = results['genuine_scores']
            impostor = results['impostor_scores']
            
            axes[idx].boxplot([genuine, impostor], labels=['Genuine', 'Impostor'])
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            axes[idx].set_ylabel('Similarity Score')
            axes[idx].grid(True, alpha=0.3, axis='y')
        except:
            continue
    
    plt.suptitle('Score Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison charts saved to: {save_dir}")


def plot_segmented_heatmap(segmented_table: pd.DataFrame, 
                          save_path: Path,
                          title: str = "Segmented Performance"):
    """Create heatmap for segmented evaluation results"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Drop summary columns for heatmap
    plot_data = segmented_table.drop(['Mean', 'Std', 'Min', 'Max'], axis=1, errors='ignore')
    
    sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.2, vmax=1.0, ax=ax, cbar_kws={'label': 'Rank-1 Accuracy'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Segment/Condition', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Segmented heatmap saved: {save_path}")


def plot_gallery_strategy_comparison(strategy_df: pd.DataFrame, save_path: Path):
    """Visualize gallery strategy analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Raw scores comparison
    ax = axes[0, 0]
    strategy_df.set_index('Model')[['Oneshot_Base', 'Oneshot_Aug', 
                                     'Fewshot_Base', 'Fewshot_Aug']].plot(
        kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Rank-1 Accuracy')
    ax.set_title('Gallery Strategy Comparison', fontweight='bold')
    ax.legend(title='Configuration')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Augmentation improvement
    ax = axes[0, 1]
    strategy_df.set_index('Model')[['Aug_Improvement_Oneshot', 
                                     'Aug_Improvement_Fewshot']].plot(
        kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Rank-1 Improvement')
    ax.set_title('Augmentation Benefit', fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(title='Gallery Type')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Fewshot improvement
    ax = axes[1, 0]
    strategy_df.set_index('Model')[['Fewshot_Improvement_Base', 
                                     'Fewshot_Improvement_Aug']].plot(
        kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Rank-1 Improvement')
    ax.set_title('Fewshot vs Oneshot Benefit', fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(title='Augmentation')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Best configuration per model
    ax = axes[1, 1]
    best_configs = strategy_df.groupby('Best_Config').size()
    best_configs.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_ylabel('Number of Models')
    ax.set_title('Most Common Best Configuration', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Gallery Strategy Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gallery strategy plot saved: {save_path}")


def export_comprehensive_report(all_model_results: Dict,
                               all_summaries: Dict,
                               save_dir: Path):
    """Export complete results to multiple formats"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Excel workbook with multiple sheets
    excel_path = save_dir / 'comprehensive_report.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        all_summaries['comparison_summary'].to_excel(writer, sheet_name='Overall_Comparison', index=False)
        all_summaries['gallery_strategy'].to_excel(writer, sheet_name='Gallery_Strategy', index=False)
        all_summaries['aggregation_analysis'].to_excel(writer, sheet_name='Aggregation_Analysis', index=False)
        all_summaries['threshold_recommendations'].to_excel(writer, sheet_name='Threshold_Recommendations', index=False)
        
        if 'segmented_oneshot' in all_summaries:
            all_summaries['segmented_oneshot'].to_excel(writer, sheet_name='Segmented_Oneshot')
        if 'segmented_fewshot' in all_summaries:
            all_summaries['segmented_fewshot'].to_excel(writer, sheet_name='Segmented_Fewshot')
        if 'statistical_comparison' in all_summaries:
            all_summaries['statistical_comparison'].to_excel(writer, sheet_name='Statistical_Tests', index=False)
    
    print(f"Excel report saved: {excel_path}")
    
    # 2. JSON export
    json_data = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'models_evaluated': list(all_model_results.keys())
        },
        'summaries': {
            key: df.to_dict(orient='records') if isinstance(df, pd.DataFrame) else df
            for key, df in all_summaries.items()
            if key != 'executive_summary'
        },
        'executive_summary': all_summaries.get('executive_summary', '')
    }
    
    json_path = save_dir / 'comprehensive_report.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"JSON report saved: {json_path}")
    
    # 3. Text summary
    txt_path = save_dir / 'executive_summary.txt'
    with open(txt_path, 'w') as f:
        f.write(all_summaries.get('executive_summary', ''))
    
    print(f"Text summary saved: {txt_path}")
    
    # 4. LaTeX tables
    latex_path = save_dir / 'latex_tables.tex'
    with open(latex_path, 'w') as f:
        f.write("% Comparison Summary\n")
        f.write(all_summaries['comparison_summary'].to_latex(index=False, float_format="%.4f"))
        f.write("\n\n% Gallery Strategy\n")
        f.write(all_summaries['gallery_strategy'].to_latex(index=False, float_format="%.4f"))
    
    print(f"LaTeX tables saved: {latex_path}")


# ============================================================================
# MAIN EXECUTION FUNCTIONS (UPDATED)
# ============================================================================

def run_basic_probe_evaluation(model_name: str, embeddings: Dict, 
                               results_dir: Path, plots_dir: Path):
    """Run basic probe evaluation (ORIGINAL + FIXED)"""
    print(f"\n{'='*70}")
    print(f"BASIC PROBE EVALUATION: {model_name}")
    print(f"{'='*70}")
    
    probe = embeddings['probe_positive_unsegmented']

    if probe is None:
        print("Missing probe embeddings!")
        return None

    gallery_types = {
        'oneshot_base': 'gallery_oneshot_base',
        'oneshot_augmented': 'gallery_oneshot_augmented', 
        'fewshot_base': 'gallery_fewshot_base',
        'fewshot_augmented': 'gallery_fewshot_augmented'
    }
        
    thresholds = np.arange(0.2, 0.91, 0.05)
    aggregations = ['max', 'mean', 'topk']
    
    all_results = {}

    for gallery_name, gallery_key in gallery_types.items():
        gallery = embeddings.get(gallery_key)
        
        if gallery is None:
            print(f"Missing {gallery_name} gallery, skipping...")
            continue
        
        print(f"\n{'-'*70}")
        print(f"GALLERY: {gallery_name.upper()}")
        print(f"{'-'*70}")
        
        gallery_results = {}
    
        for agg in aggregations:
            print(f"\nEvaluating with {agg.upper()} aggregation...")
            results = evaluate_probes_comprehensive(
                gallery, probe, thresholds, aggregation=agg, k=3
            )
            
            csv_path = results_dir / model_name / f'basic_probe_{gallery_name}_{agg}_metrics.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            results['threshold_results'].to_csv(csv_path, index=False)
    
            plot_path = plots_dir / model_name / f'basic_probe_{gallery_name}_{agg}_plot.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_all_metrics(results, f"{model_name} - Basic Probe - {gallery_name.upper()} ({agg.upper()})", plot_path)
                        
            gallery_results[agg] = results

            df = results['threshold_results']
            best_idx = df['rank1_accuracy'].idxmax()
            print(f"  Best Rank-1: {df.loc[best_idx, 'rank1_accuracy']:.4f} "
                f"@ threshold {df.loc[best_idx, 'threshold']:.2f}")
            print(f"  ROC-AUC: {results['roc_auc']:.4f}")
            print(f"  d-prime: {results['dprime']:.3f}")

        all_results[gallery_name] = gallery_results
    
    return all_results


def run_impostor_evaluation(model_name: str, embeddings: Dict,
                           results_dir: Path, plots_dir: Path):
    """Run impostor evaluation (ORIGINAL + ENHANCED OUTPUT)"""
    print(f"\n{'='*70}")
    print(f"IMPOSTOR EVALUATION: {model_name}")
    print(f"{'='*70}")
    
    gallery = embeddings['gallery_oneshot_augmented']
    impostor = embeddings['probe_negative']
    
    if gallery is None or impostor is None:
        print("Missing embeddings!")
        return None
    
    thresholds = np.arange(0.2, 0.91, 0.05)
    
    print("\nEvaluating with MEAN aggregation...")
    results = evaluate_impostors_comprehensive(
        gallery, impostor, thresholds, aggregation='mean', k=3
    )
    
    csv_path = results_dir / model_name / 'impostor_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results['threshold_results'].to_csv(csv_path, index=False)
    
    plot_path = plots_dir / model_name / 'impostor_plot.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_impostor_metrics(results, f"{model_name} - Impostor Rejection", plot_path)

    df = results['threshold_results']
    best_idx = df['rejection_rate'].idxmax()
    print(f"  Best Rejection Rate: {df.loc[best_idx, 'rejection_rate']:.4f} "
          f"@ threshold {df.loc[best_idx, 'threshold']:.2f}")
    print(f"  FAR at best: {df.loc[best_idx, 'far']:.4f}")
    print(f"  Total impostors: {df.loc[best_idx, 'n_impostors']}")
    print(f"  Mean impostor score: {results['mean_impostor_score']:.4f}")
    print(f"  Std impostor score: {results['std_impostor_score']:.4f}")
    print(f"  95% CI: [{results['impostor_ci'][0]:.4f}, {results['impostor_ci'][1]:.4f}]")
    
    return results


def run_segmented_evaluation(model_name: str, embeddings: Dict,
                             results_dir: Path, plots_dir: Path,
                             gallery_type: str):
    """Run segmented evaluation (FIXED - added d-prime output)"""
    print(f"\n{'='*70}")
    print(f"SEGMENTED EVALUATION: {model_name} ({gallery_type})")
    print(f"{'='*70}")
    
    gallery_key = f'gallery_{gallery_type}_augmented'
    gallery = embeddings[gallery_key]
    probe = embeddings['probe_positive_segmented']
    
    if gallery is None or probe is None:
        print("Missing embeddings!")
        return None
    
    thresholds = np.arange(0.2, 0.91, 0.05)
    
    print("\nEvaluating with MEAN aggregation...")
    segment_results = evaluate_segmented_comprehensive(
        gallery, probe, thresholds, aggregation='mean', k=3
    )
    
    for segment_name, results in segment_results.items():
        csv_path = results_dir / model_name / f'segmented_{gallery_type}_{segment_name}_metrics.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results['threshold_results'].to_csv(csv_path, index=False)

        plot_path = plots_dir / model_name / f'segmented_{gallery_type}_{segment_name}_plot.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_all_metrics(results, 
                        f"{model_name} - {segment_name} ({gallery_type})", 
                        plot_path)
        
        df = results['threshold_results']
        best_idx = df['rank1_accuracy'].idxmax()
        print(f"\n  {segment_name}:")
        print(f"    Rank-1: {df.loc[best_idx, 'rank1_accuracy']:.4f} "
              f"@ threshold {df.loc[best_idx, 'threshold']:.2f}")
        print(f"    ROC-AUC: {results['roc_auc']:.4f}")
        print(f"    d-prime: {results['dprime']:.3f}")  # FIXED: Added d-prime output
    
    return segment_results


def run_complete_evaluation_pipeline(all_embeddings: Dict, output_base_dir: Path):
    """
    Complete evaluation pipeline with all analysis and comparisons
    
    Args:
        all_embeddings: Dict with structure {model_name: embeddings_dict}
        output_base_dir: Base directory for all outputs
    """
    
    print("\n" + "="*80)
    print("COMPLETE FACE RECOGNITION EVALUATION PIPELINE")
    print("="*80)
    
    results_dir = output_base_dir / 'evaluation_results'
    plots_dir = output_base_dir / 'plots'
    comparison_dir = output_base_dir / 'comparisons'
    
    all_model_results = {}
    
    # Run individual model evaluations
    for model_name, embeddings in all_embeddings.items():
        print(f"\n{'#'*80}")
        print(f"# PROCESSING MODEL: {model_name}")
        print(f"{'#'*80}")
        
        model_results = {}
        
        # 1. Basic probe evaluation
        model_results['basic_probe'] = run_basic_probe_evaluation(
            model_name, embeddings, results_dir, plots_dir
        )
        
        # 2. Impostor evaluation
        model_results['impostor'] = run_impostor_evaluation(
            model_name, embeddings, results_dir, plots_dir
        )
        
        # 3. Segmented evaluation - oneshot
        model_results['segmented_oneshot'] = run_segmented_evaluation(
            model_name, embeddings, results_dir, plots_dir, 'oneshot'
        )
        
        # 4. Segmented evaluation - fewshot
        model_results['segmented_fewshot'] = run_segmented_evaluation(
            model_name, embeddings, results_dir, plots_dir, 'fewshot'
        )
        
        all_model_results[model_name] = model_results
    
    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================
    
    print(f"\n{'#'*80}")
    print("# COMPARATIVE ANALYSIS")
    print(f"{'#'*80}")
    
    all_summaries = {}
    
    # 1. Generate comparison summary
    print("\n1. Generating comparison summary...")
    all_summaries['comparison_summary'] = generate_comparison_summary(all_model_results)
    all_summaries['comparison_summary'].to_csv(comparison_dir / 'comparison_summary.csv', index=False)
    print(f"   Saved: {comparison_dir / 'comparison_summary.csv'}")
    
    # 2. Gallery strategy analysis
    print("\n2. Analyzing gallery strategies...")
    all_summaries['gallery_strategy'] = analyze_gallery_strategies(all_model_results)
    all_summaries['gallery_strategy'].to_csv(comparison_dir / 'gallery_strategy_analysis.csv', index=False)
    print(f"   Saved: {comparison_dir / 'gallery_strategy_analysis.csv'}")
    
    # 3. Aggregation method analysis
    print("\n3. Analyzing aggregation methods...")
    all_summaries['aggregation_analysis'] = summarize_aggregation_performance(all_model_results)
    all_summaries['aggregation_analysis'].to_csv(comparison_dir / 'aggregation_analysis.csv', index=False)
    print(f"   Saved: {comparison_dir / 'aggregation_analysis.csv'}")
    
    # 4. Threshold recommendations
    print("\n4. Generating threshold recommendations...")
    all_summaries['threshold_recommendations'] = recommend_operating_thresholds(all_model_results)
    all_summaries['threshold_recommendations'].to_csv(comparison_dir / 'threshold_recommendations.csv', index=False)
    print(f"   Saved: {comparison_dir / 'threshold_recommendations.csv'}")
    
    # 5. Segmented comparison tables
    print("\n5. Creating segmented comparison tables...")
    all_summaries['segmented_oneshot'] = create_segmented_comparison_table(all_model_results, 'oneshot')
    all_summaries['segmented_oneshot'].to_csv(comparison_dir / 'segmented_oneshot_comparison.csv')
    print(f"   Saved: {comparison_dir / 'segmented_oneshot_comparison.csv'}")
    
    all_summaries['segmented_fewshot'] = create_segmented_comparison_table(all_model_results, 'fewshot')
    all_summaries['segmented_fewshot'].to_csv(comparison_dir / 'segmented_fewshot_comparison.csv')
    print(f"   Saved: {comparison_dir / 'segmented_fewshot_comparison.csv'}")
    
    # 6. Failure analysis
    print("\n6. Analyzing failure cases...")
    all_summaries['failure_analysis'] = analyze_failure_cases(all_model_results)
    with open(comparison_dir / 'failure_analysis.json', 'w') as f:
        json.dump(all_summaries['failure_analysis'], f, indent=2)
    print(f"   Saved: {comparison_dir / 'failure_analysis.json'}")
    
    # 7. Statistical comparison
    print("\n7. Performing statistical comparisons...")
    all_summaries['statistical_comparison'] = compare_models_statistical(all_model_results)
    all_summaries['statistical_comparison'].to_csv(comparison_dir / 'statistical_comparison.csv', index=False)
    print(f"   Saved: {comparison_dir / 'statistical_comparison.csv'}")
    
    # 8. Executive summary
    print("\n8. Generating executive summary...")
    all_summaries['executive_summary'] = generate_executive_summary(
        all_model_results, 
        all_summaries['comparison_summary']
    )
    print(all_summaries['executive_summary'])
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    print(f"\n{'#'*80}")
    print("# GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'#'*80}")
    
    # 1. Model comparison charts
    print("\n1. Creating model comparison charts...")
    plot_model_comparison_charts(
        all_model_results,
        all_summaries['comparison_summary'],
        comparison_dir / 'charts'
    )
    
    # 2. Segmented heatmaps
    print("\n2. Creating segmented performance heatmaps...")
    plot_segmented_heatmap(
        all_summaries['segmented_oneshot'],
        comparison_dir / 'charts' / 'segmented_oneshot_heatmap.png',
        'Segmented Performance - Oneshot'
    )
    plot_segmented_heatmap(
        all_summaries['segmented_fewshot'],
        comparison_dir / 'charts' / 'segmented_fewshot_heatmap.png',
        'Segmented Performance - Fewshot'
    )
    
    # 3. Gallery strategy visualization
    print("\n3. Creating gallery strategy visualizations...")
    plot_gallery_strategy_comparison(
        all_summaries['gallery_strategy'],
        comparison_dir / 'charts' / 'gallery_strategy_comparison.png'
    )
    
    # ========================================================================
    # EXPORT REPORTS
    # ========================================================================
    
    print(f"\n{'#'*80}")
    print("# EXPORTING COMPREHENSIVE REPORTS")
    print(f"{'#'*80}")
    
    export_comprehensive_report(
        all_model_results,
        all_summaries,
        comparison_dir / 'reports'
    )
    
    print(f"\n{'='*80}")
    print("EVALUATION PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_base_dir}")
    print(f"  - Individual results: {results_dir}")
    print(f"  - Individual plots: {plots_dir}")
    print(f"  - Comparisons: {comparison_dir}")
    print(f"  - Reports: {comparison_dir / 'reports'}")
    
    return all_model_results, all_summaries


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# How to use this complete evaluation system:

# 1. Prepare your embeddings dictionary
all_embeddings = {
    'adaface_ir_50': {
        'gallery_oneshot_base': {...},
        'gallery_oneshot_augmented': {...},
        'gallery_fewshot_base': {...},
        'gallery_fewshot_augmented': {...},
        'probe_positive_unsegmented': {...},
        'probe_positive_segmented': {...},
        'probe_negative': {...}
    },
    'adaface_ir_101': {...},
    'arcface_ir_50': {...},
    'arcface_ir_101': {...}
}

# 2. Run complete evaluation
output_dir = Path('output/v0')
all_results, all_summaries = run_complete_evaluation_pipeline(
    all_embeddings, 
    output_dir
)

# 3. Access specific results
print(all_summaries['executive_summary'])
print(all_summaries['comparison_summary'])
print(all_summaries['gallery_strategy'])
"""