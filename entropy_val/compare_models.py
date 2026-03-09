"""
Model Comparison Script
Compare two trained models on entropy validation metrics
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def load_results(model_dir):
    """Load all test results from a model directory"""
    results = {}
    
    # Load individual test stats
    for test_name in ['darkness', 'distance', 'occlusion', 'complexity', 'ensemble_disagreement']:
        json_file = os.path.join(model_dir, f'{test_name}_stats.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results[test_name] = json.load(f)
    
    # Load summary
    summary_file = os.path.join(model_dir, 'summary_report.txt')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            results['summary'] = f.read()
    
    return results

def parse_model_name(model_dir):
    """Extract readable model name from directory"""
    # e.g., "entropy_validation_results_hybrid_entropy_new_2025-12-27_22-09-42"
    parts = model_dir.replace('entropy_validation_results_', '').split('_')
    
    # Try to extract meaningful name
    if len(parts) >= 2:
        return '_'.join(parts[:2])  # e.g., "hybrid_entropy"
    return os.path.basename(model_dir)

def compare_two_models(model1_dir, model2_dir, output_dir='model_comparison'):
    """Generate comprehensive comparison report"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Load results
    results1 = load_results(model1_dir)
    results2 = load_results(model2_dir)
    
    model1_name = parse_model_name(model1_dir)
    model2_name = parse_model_name(model2_dir)
    
    print(f"\nModel 1: {model1_name}")
    print(f"Model 2: {model2_name}")
    print()
    
    # ===== TEST 1: DARKNESS =====
    if 'darkness' in results1 and 'darkness' in results2:
        print("TEST 1: DARKNESS SENSITIVITY")
        print("-" * 80)
        
        table_data = [
            ['Metric', model1_name, model2_name, 'Winner'],
            ['Ensemble Correlation', 
             f"{results1['darkness']['ensemble_correlation']:.3f}", 
             f"{results2['darkness']['ensemble_correlation']:.3f}",
             '✓ Model 2' if abs(results2['darkness']['ensemble_correlation']) > abs(results1['darkness']['ensemble_correlation']) else '✓ Model 1'],
            ['Ensemble P-value', 
             f"{results1['darkness']['ensemble_p_value']:.4f}", 
             f"{results2['darkness']['ensemble_p_value']:.4f}",
             '✓ Model 2' if results2['darkness']['ensemble_p_value'] < results1['darkness']['ensemble_p_value'] else '✓ Model 1'],
            ['Dark Scene Entropy', 
             f"{results1['darkness']['dark_ensemble_mean']:.4f}", 
             f"{results2['darkness']['dark_ensemble_mean']:.4f}",
             '✓ Model 2' if results2['darkness']['dark_ensemble_mean'] > results1['darkness']['dark_ensemble_mean'] else '✓ Model 1'],
        ]
        print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
        print()
    
    # ===== TEST 2: DISTANCE =====
    if 'distance' in results1 and 'distance' in results2:
        print("TEST 2: DISTANCE-BASED UNCERTAINTY")
        print("-" * 80)
        
        table_data = [
            ['Metric', model1_name, model2_name, 'Winner'],
            ['ANOVA F-statistic', 
             f"{results1['distance']['ensemble_anova_f']:.2f}", 
             f"{results2['distance']['ensemble_anova_f']:.2f}",
             '✓ Model 2' if results2['distance']['ensemble_anova_f'] > results1['distance']['ensemble_anova_f'] else '✓ Model 1'],
            ['Near (0-15m) Entropy', 
             f"{results1['distance']['ensemble_means'][0]:.4f}", 
             f"{results2['distance']['ensemble_means'][0]:.4f}",
             '-'],
            ['Far (30-45m) Entropy', 
             f"{results1['distance']['ensemble_means'][2]:.4f}", 
             f"{results2['distance']['ensemble_means'][2]:.4f}",
             '-'],
            ['Increases with distance?', 
             '✓' if results1['distance']['ensemble_means'][2] > results1['distance']['ensemble_means'][0] else '✗',
             '✓' if results2['distance']['ensemble_means'][2] > results2['distance']['ensemble_means'][0] else '✗',
             '✓ Model 2' if (results2['distance']['ensemble_means'][2] > results2['distance']['ensemble_means'][0]) else '✓ Model 1'],
        ]
        print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
        print()
    
    # ===== TEST 3: OCCLUSION =====
    if 'occlusion' in results1 and 'occlusion' in results2:
        print("TEST 3: OCCLUSION RESPONSE")
        print("-" * 80)
        
        table_data = [
            ['Metric', model1_name, model2_name, 'Winner'],
            ['Mean Entropy Change', 
             f"{results1['occlusion']['ensemble_mean_delta']:.4f}", 
             f"{results2['occlusion']['ensemble_mean_delta']:.4f}",
             '✓ Model 2' if results2['occlusion']['ensemble_mean_delta'] > results1['occlusion']['ensemble_mean_delta'] else '✓ Model 1'],
            ['P-value', 
             f"{results1['occlusion']['ensemble_ttest_pvalue']:.4f}", 
             f"{results2['occlusion']['ensemble_ttest_pvalue']:.4f}",
             '✓ Model 2' if results2['occlusion']['ensemble_ttest_pvalue'] < results1['occlusion']['ensemble_ttest_pvalue'] else '✓ Model 1'],
            ['Significant (p<0.05)?', 
             '✓' if results1['occlusion']['ensemble_ttest_pvalue'] < 0.05 else '✗',
             '✓' if results2['occlusion']['ensemble_ttest_pvalue'] < 0.05 else '✗',
             '✓ Model 2' if results2['occlusion']['ensemble_ttest_pvalue'] < 0.05 else '✓ Model 1'],
        ]
        print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
        print()
    
    # ===== TEST 4: COMPLEXITY =====
    if 'complexity' in results1 and 'complexity' in results2:
        print("TEST 4: SCENE COMPLEXITY CORRELATION")
        print("-" * 80)
        
        table_data = [
            ['Metric', model1_name, model2_name, 'Winner'],
            ['Ensemble Correlation', 
             f"{results1['complexity']['ensemble_correlation']:.3f}", 
             f"{results2['complexity']['ensemble_correlation']:.3f}",
             '✓ Model 2' if results2['complexity']['ensemble_correlation'] > results1['complexity']['ensemble_correlation'] else '✓ Model 1'],
            ['P-value', 
             f"{results1['complexity']['ensemble_p_value']:.4f}", 
             f"{results2['complexity']['ensemble_p_value']:.4f}",
             '✓ Model 2' if results2['complexity']['ensemble_p_value'] < results1['complexity']['ensemble_p_value'] else '✓ Model 1'],
            ['In ideal range (0.3-0.7)?', 
             '✓' if 0.3 <= results1['complexity']['ensemble_correlation'] <= 0.7 else '✗',
             '✓' if 0.3 <= results2['complexity']['ensemble_correlation'] <= 0.7 else '✗',
             '-'],
        ]
        print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
        print()
    
    # ===== TEST 5: DISAGREEMENT =====
    if 'ensemble_disagreement' in results1 and 'ensemble_disagreement' in results2:
        print("TEST 5: ENSEMBLE DISAGREEMENT")
        print("-" * 80)
        
        table_data = [
            ['Metric', model1_name, model2_name, 'Winner'],
            ['Variance-Entropy Correlation', 
             f"{results1['ensemble_disagreement']['correlation']:.3f}", 
             f"{results2['ensemble_disagreement']['correlation']:.3f}",
             '✓ Model 2' if results2['ensemble_disagreement']['correlation'] > results1['ensemble_disagreement']['correlation'] else '✓ Model 1'],
            ['Meets target (>0.7)?', 
             '✓' if results1['ensemble_disagreement']['correlation'] > 0.7 else '✗',
             '✓' if results2['ensemble_disagreement']['correlation'] > 0.7 else '✗',
             '✓ Model 2' if results2['ensemble_disagreement']['correlation'] > 0.7 else '✓ Model 1'],
        ]
        print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
        print()
    
    # ===== OVERALL SCORE =====
    print("=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    score1 = calculate_score(results1)
    score2 = calculate_score(results2)
    
    print(f"\n{model1_name}: {score1}/5 tests passed")
    print(f"{model2_name}: {score2}/5 tests passed")
    print()
    
    if score2 > score1:
        print(f"🏆 WINNER: {model2_name} (+{score2-score1} tests)")
        print(f"✅ RECOMMENDATION: Use {model2_name} for radar simulation")
    elif score1 > score2:
        print(f"🏆 WINNER: {model1_name} (+{score1-score2} tests)")
        print(f"⚠️  RECOMMENDATION: {model2_name} performed worse, keep using {model1_name}")
    else:
        print(f"🤝 TIE: Both models pass {score1}/5 tests")
        print("   Check individual metrics for nuanced differences")
    
    print("=" * 80)
    
    # Save comparison report
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write(f"Model 1: {model1_name} - Score: {score1}/5\n")
        f.write(f"Model 2: {model2_name} - Score: {score2}/5\n")
        f.write(f"\nWinner: {model2_name if score2 > score1 else model1_name}\n")

def calculate_score(results):
    """Calculate how many tests a model passes"""
    score = 0
    
    # Test 1: Darkness (pass if significant negative correlation)
    if 'darkness' in results:
        if results['darkness']['ensemble_p_value'] < 0.05 and results['darkness']['ensemble_correlation'] < -0.25:
            score += 1
    
    # Test 2: Distance (pass if increasing with distance)
    if 'distance' in results:
        means = results['distance']['ensemble_means']
        if means[2] > means[0]:  # Far > Near
            score += 1
    
    # Test 3: Occlusion (pass if significant positive increase)
    if 'occlusion' in results:
        if results['occlusion']['ensemble_ttest_pvalue'] < 0.05 and results['occlusion']['ensemble_mean_delta'] > 0:
            score += 1
    
    # Test 4: Complexity (pass if moderate correlation)
    if 'complexity' in results:
        corr = results['complexity']['ensemble_correlation']
        if 0.3 <= corr <= 0.7 and results['complexity']['ensemble_p_value'] < 0.05:
            score += 1
    
    # Test 5: Disagreement (pass if correlation > 0.7)
    if 'ensemble_disagreement' in results:
        if results['ensemble_disagreement']['correlation'] > 0.7:
            score += 1
    
    return score

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py <model1_dir> <model2_dir> [output_dir]")
        print("\nExample:")
        print("  python compare_models.py \\")
        print("      entropy_validation_results_old_model \\")
        print("      entropy_validation_results_new_model")
        sys.exit(1)
    
    model1_dir = sys.argv[1]
    model2_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'model_comparison'
    
    if not os.path.exists(model1_dir):
        print(f"Error: {model1_dir} not found")
        sys.exit(1)
    
    if not os.path.exists(model2_dir):
        print(f"Error: {model2_dir} not found")
        sys.exit(1)
    
    compare_two_models(model1_dir, model2_dir, output_dir)

if __name__ == "__main__":
    main()

