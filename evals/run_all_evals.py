"""
Run All Evaluations
===================
Master script to run all evaluation scripts and generate a combined report.
"""

import json
from pathlib import Path
from datetime import datetime

from eval_data_pipeline import run_evaluation as run_data_pipeline_eval
from eval_feature_engineering import run_evaluation as run_feature_engineering_eval
from eval_tensor_preparation import run_evaluation as run_tensor_preparation_eval


def run_all_evaluations():
    """Run all evaluation scripts and combine results."""
    project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("RETAILSIM COMPLETE EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = {}
    section_scores = []

    # Section 2: Data Pipeline
    print("\n" + "=" * 70)
    print("SECTION 2: DATA PIPELINE")
    print("=" * 70)
    try:
        results = run_data_pipeline_eval(project_root)
        all_results['section2_data_pipeline'] = results
        section_scores.append(results['overall']['quality_score'])
    except Exception as e:
        print(f"Error running data pipeline eval: {e}")
        all_results['section2_data_pipeline'] = {'error': str(e), 'quality_score': 0}
        section_scores.append(0)

    # Section 3: Feature Engineering
    print("\n" + "=" * 70)
    print("SECTION 3: FEATURE ENGINEERING")
    print("=" * 70)
    try:
        results = run_feature_engineering_eval(project_root)
        all_results['section3_feature_engineering'] = results
        section_scores.append(results['overall']['quality_score'])
    except Exception as e:
        print(f"Error running feature engineering eval: {e}")
        all_results['section3_feature_engineering'] = {'error': str(e), 'quality_score': 0}
        section_scores.append(0)

    # Section 4: Tensor Preparation
    print("\n" + "=" * 70)
    print("SECTION 4: TENSOR PREPARATION")
    print("=" * 70)
    try:
        results = run_tensor_preparation_eval(project_root)
        all_results['section4_tensor_preparation'] = results
        section_scores.append(results['overall']['quality_score'])
    except Exception as e:
        print(f"Error running tensor preparation eval: {e}")
        all_results['section4_tensor_preparation'] = {'error': str(e), 'quality_score': 0}
        section_scores.append(0)

    # Summary
    overall_score = sum(section_scores) / len(section_scores) if section_scores else 0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n  Section 2 (Data Pipeline):       {section_scores[0]:.1f}/100")
    print(f"  Section 3 (Feature Engineering): {section_scores[1]:.1f}/100")
    print(f"  Section 4 (Tensor Preparation):  {section_scores[2]:.1f}/100")
    print(f"\n  OVERALL SCORE: {overall_score:.1f}/100")
    print("=" * 70)

    all_results['summary'] = {
        'timestamp': datetime.now().isoformat(),
        'section_scores': {
            'section2_data_pipeline': section_scores[0],
            'section3_feature_engineering': section_scores[1],
            'section4_tensor_preparation': section_scores[2],
        },
        'overall_score': overall_score,
    }

    # Save combined results
    output_path = project_root / 'evals' / 'all_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nCombined results saved to: {output_path}")

    return all_results


if __name__ == '__main__':
    run_all_evaluations()
