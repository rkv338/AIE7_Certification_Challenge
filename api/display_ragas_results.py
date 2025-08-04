#!/usr/bin/env python3
"""
📊 RAGAS Results Display Script

Reads the RAGAS evaluation results JSON and displays a nice table format.
"""

import json
import pandas as pd
from datetime import datetime

def load_and_display_results(json_file: str = "ragas_evaluation_results_fixed.json"):
    """Load and display RAGAS results in table format."""
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("🎯 RAGAS EVALUATION RESULTS")
        print("=" * 80)
        
        # Display dataset info
        dataset_info = results.get("dataset_info", {})
        print(f"\n📋 Dataset Information:")
        print(f"   File: {dataset_info.get('dataset_path', 'N/A')}")
        print(f"   Total Questions: {dataset_info.get('total_questions', 'N/A')}")
        print(f"   Evaluation Date: {results.get('evaluation_timestamp', 'N/A')}")
        print(f"   Status: {'✅ Completed' if dataset_info.get('evaluation_completed', False) else '❌ Failed'}")
        
        # Display overall metrics if available
        if "overall_metrics" in results and results["overall_metrics"]:
            print(f"\n📊 OVERALL METRICS:")
            print("-" * 50)
            metrics_table = []
            for metric, score in results["overall_metrics"].items():
                emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.7 else "🔴"
                metric_name = metric.replace('_', ' ').title()
                metrics_table.append({
                    "Metric": f"{emoji} {metric_name}",
                    "Score": f"{score:.3f}"
                })
            
            df_metrics = pd.DataFrame(metrics_table)
            print(df_metrics.to_string(index=False))
            
            # Performance analysis
            avg_score = sum(results["overall_metrics"].values()) / len(results["overall_metrics"])
            print(f"\n🎯 Overall Performance: {avg_score:.3f}")
            if avg_score >= 0.8:
                print("   Excellent! Your agent is performing very well.")
            elif avg_score >= 0.7:
                print("   Good performance with room for improvement.")
            else:
                print("   Needs improvement - consider reviewing prompts and context retrieval.")
        
        # Display detailed results table
        if "detailed_results" in results and results["detailed_results"]:
            print(f"\n📋 DETAILED RESULTS:")
            print("-" * 80)
            
            detailed_data = []
            for result in results["detailed_results"]:
                metrics = result.get("metrics", {})
                detailed_data.append({
                    "ID": result.get("question_id", "N/A"),
                    "Question": result.get("question", "")[:40] + "...",
                    "Faithfulness": f"{metrics.get('faithfulness', 0):.3f}" if metrics.get('faithfulness') is not None else "N/A",
                    "Relevancy": f"{metrics.get('answer_relevancy', 0):.3f}" if metrics.get('answer_relevancy') is not None else "N/A",
                    "Context Precision": f"{metrics.get('context_precision', 0):.3f}" if metrics.get('context_precision') is not None else "N/A",
                    "Context Recall": f"{metrics.get('context_recall', 0):.3f}" if metrics.get('context_recall') is not None else "N/A"
                })
            
            df_detailed = pd.DataFrame(detailed_data)
            print(df_detailed.to_string(index=False))
        
        # Show error info if evaluation failed
        if not dataset_info.get('evaluation_completed', False) and 'error' in dataset_info:
            print(f"\n❌ EVALUATION ERROR:")
            print(f"   {dataset_info['error']}")
            
            # Show raw responses if available
            if "raw_responses" in results:
                print(f"\n📝 Raw Responses Available: {len(results['raw_responses'])} questions")
                print("   Use the raw responses to debug context extraction issues.")
        
        return results
        
    except FileNotFoundError:
        print(f"❌ File not found: {json_file}")
        print("   Run the evaluation script first: python fixed_ragas_evaluation.py")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {json_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def compare_with_baseline():
    """Compare results with a baseline performance."""
    
    baseline_metrics = {
        "faithfulness": 0.75,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.65
    }
    
    results = load_and_display_results()
    if not results or "overall_metrics" not in results:
        return
    
    print(f"\n📈 COMPARISON WITH BASELINE:")
    print("-" * 40)
    
    for metric, baseline in baseline_metrics.items():
        actual = results["overall_metrics"].get(metric, 0)
        diff = actual - baseline
        status = "📈" if diff > 0 else "📉" if diff < 0 else "📊"
        
        print(f"{status} {metric.replace('_', ' ').title():<18}: {actual:.3f} (baseline: {baseline:.3f}, diff: {diff:+.3f})")

if __name__ == "__main__":
    print("🚀 Loading RAGAS Evaluation Results\n")
    results = load_and_display_results()
    
    if results:
        print("\n" + "="*80)
        compare_with_baseline()
        print("\n💡 To improve performance:")
        print("   - Enhance context extraction from agent responses")
        print("   - Improve prompt engineering for more grounded responses")
        print("   - Ensure RAG retrieval is finding relevant information")
    
    print("\n✅ Analysis complete!")