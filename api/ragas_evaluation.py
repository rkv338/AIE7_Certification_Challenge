#!/usr/bin/env python3
"""
🎯 RAGAS Agent Evaluation Script - FIXED VERSION

This script evaluates your therapy Agent using RAGAS metrics with your dataset.csv file.
Outputs comprehensive evaluation results to a JSON file.

Usage: python fixed_ragas_evaluation.py
"""

import asyncio
import json
from uuid import uuid4
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any
import os
import io
import time
from dotenv import load_dotenv

# Import Agent components
from Agent import create_agent_graph, set_therapy_vector_db
from app import upload_pdf_rag
from langchain_core.messages import HumanMessage
from fastapi import UploadFile

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentRAGASEvaluator:
    def __init__(self, dataset_path: str = "dataset.csv"):
        """Initialize the evaluator with dataset path."""
        self.dataset_path = dataset_path
        self.agent_graph = None
        self.results = []
        
    async def setup_agent(self):
        """Set up the therapy agent with vector database."""
        try:
            logger.info("🚀 Setting up therapy agent...")
            
            # Upload PDF and set up vector database
            with open("../docs/Therapy Doc.pdf", "rb") as file:
                file_like = io.BytesIO(file.read())
                file_like.name = "Therapy Doc.pdf"
                pdf_id = await upload_pdf_rag(file=UploadFile(file=file_like))
                logger.info(f"📚 PDF uploaded with ID: {pdf_id}")
            
            # Small delay to ensure setup is complete
            await asyncio.sleep(2)
            
            # Create agent graph
            self.agent_graph = create_agent_graph()
            logger.info("✅ Agent setup complete!")
            
        except Exception as e:
            logger.error(f"💥 Agent setup failed: {e}")
            raise
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the test dataset from CSV."""
        try:
            df = pd.read_csv(self.dataset_path)
            logger.info(f"📊 Loaded dataset with {len(df)} test cases")
            return df
        except Exception as e:
            logger.error(f"💥 Failed to load dataset: {e}")
            raise
    
    async def get_agent_response(self, question: str) -> Dict[str, Any]:
        """Get response from agent including contexts."""
        try:
            inputs = {"messages": [HumanMessage(content=question)]}
            
            # Get full response from agent
            response = await self.agent_graph.ainvoke(inputs)
            
            # Extract the final answer
            final_answer = response["messages"][-1].content if response["messages"] else "No response"
            
            # For RAGAS, we need to extract contexts
            contexts = []
            
            # Look through ALL messages for tool responses
            for message in response["messages"]:
                logger.info(f"🔍 Message type: {type(message).__name__}")
                logger.info(f"📝 Message content preview: {str(message)[:200]}...")
                
                # Check for ToolMessage (actual tool responses)
                if hasattr(message, 'type') and message.type == 'tool':
                    logger.info("🛠️ Found ToolMessage")
                    if hasattr(message, 'content') and isinstance(message.content, str):
                        tool_content = message.content
                        logger.info(f"🛠️ Tool content: {tool_content[:100]}...")
                        
                        # Extract context from therapy RAG tool responses
                        if "Found relevant information in therapy document:" in tool_content:
                            # Extract everything after the prefix
                            context_start = tool_content.find("Found relevant information in therapy document:") + len("Found relevant information in therapy document:")
                            extracted_context = tool_content[context_start:].strip()
                            if extracted_context and len(extracted_context) > 50:
                                contexts.append(extracted_context)
                                logger.info(f"✅ Extracted RAG context: {extracted_context[:100]}...")
                        
                        # Also check for Tavily responses
                        elif any(phrase in tool_content.lower() for phrase in ["search results", "research findings", "according to"]):
                            if len(tool_content) > 50:
                                contexts.append(tool_content)
                                logger.info(f"✅ Extracted Tavily context: {tool_content[:100]}...")
                
                # Also check regular message content for embedded context
                elif hasattr(message, 'content') and isinstance(message.content, str):
                    content = message.content
                    
                    # Look for embedded context patterns
                    if "Found relevant information in therapy document:" in content:
                        context_start = content.find("Found relevant information in therapy document:") + len("Found relevant information in therapy document:")
                        extracted_context = content[context_start:].strip()
                        if extracted_context and len(extracted_context) > 50:
                            contexts.append(extracted_context)
                            logger.info(f"✅ Extracted embedded RAG context: {extracted_context[:100]}...")
            
            # Log final context extraction results
            logger.info(f"📊 Total contexts extracted: {len(contexts)}")
            for i, ctx in enumerate(contexts):
                logger.info(f"📄 Context {i+1}: {ctx[:150]}...")
            
            # If no contexts found, create a meaningful placeholder
            if not contexts:
                logger.warning("⚠️ No contexts extracted - using placeholder")
                contexts = ["General therapeutic knowledge and empathetic guidance"]
            
            return {
                "answer": final_answer,
                "contexts": contexts
            }
            
        except Exception as e:
            logger.error(f"💥 Error getting agent response: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "contexts": ["Error retrieving context"]
            }
    
    async def evaluate_dataset(self) -> Dict[str, Any]:
        """Run evaluation on the entire dataset."""
        logger.info("🎯 Starting RAGAS evaluation...")
        
        # Setup agent
        await self.setup_agent()
        
        # Load dataset
        df = self.load_dataset()
        
        # Prepare data for RAGAS evaluation
        evaluation_data = []
        
        for index, row in df.iterrows():
            logger.info(f"📝 Evaluating question {index + 1}/{len(df)}")
            
            question = row.get('question', row.get('user_input', ''))
            ground_truth = row.get('ground_truth', row.get('reference', 'No ground truth provided'))
            
            # Get agent response
            response_data = await self.get_agent_response(question)
            
            evaluation_data.append({
                "question": question,
                "answer": response_data["answer"],
                "contexts": response_data["contexts"],
                "ground_truth": ground_truth
            })
            
            # Add delay to avoid rate limits
            await asyncio.sleep(1)
            
            # Log progress
            logger.info(f"✅ Completed: {question[:50]}...")
        
        # Convert to RAGAS dataset format
        dataset = Dataset.from_list(evaluation_data)
        
        # Run RAGAS evaluation
        logger.info("🔍 Running RAGAS metrics evaluation...")
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ]
            )
            
            # RAGAS returns a Dataset object - convert to pandas first
            result_df = result.to_pandas()
            
            # Calculate mean scores for overall metrics
            overall_metrics = {}
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in result_df.columns:
                    # Handle both single values and arrays
                    metric_values = result_df[metric]
                    if len(metric_values) > 0:
                        # If it's a series of numbers, take the mean
                        try:
                            overall_metrics[metric] = float(metric_values.mean())
                        except:
                            # If conversion fails, try to handle lists/arrays
                            overall_metrics[metric] = 0.0
                    else:
                        overall_metrics[metric] = 0.0
                else:
                    overall_metrics[metric] = 0.0
            
            # Convert results to dictionary format
            results_dict = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "dataset_info": {
                    "dataset_path": self.dataset_path,
                    "total_questions": len(df),
                    "evaluation_completed": True
                },
                "overall_metrics": overall_metrics,
                "detailed_results": []
            }
            
            # Add detailed results for each question
            for i, item in enumerate(evaluation_data):
                # Get individual metrics for this question if available
                individual_metrics = {}
                if i < len(result_df):
                    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        if metric in result_df.columns:
                            try:
                                individual_metrics[metric] = float(result_df.iloc[i][metric]) if pd.notna(result_df.iloc[i][metric]) else None
                            except:
                                individual_metrics[metric] = None
                        else:
                            individual_metrics[metric] = None
                else:
                    individual_metrics = {metric: None for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
                
                detailed_result = {
                    "question_id": i + 1,
                    "question": item["question"],
                    "agent_answer": item["answer"],
                    "ground_truth": item["ground_truth"],
                    "contexts": item["contexts"],
                    "metrics": individual_metrics
                }
                results_dict["detailed_results"].append(detailed_result)
            
            logger.info("✅ RAGAS evaluation completed successfully!")
            return results_dict
            
        except Exception as e:
            logger.error(f"💥 RAGAS evaluation failed: {e}")
            # Return error results
            return {
                "evaluation_timestamp": datetime.now().isoformat(),
                "dataset_info": {
                    "dataset_path": self.dataset_path,
                    "total_questions": len(df),
                    "evaluation_completed": False,
                    "error": str(e)
                },
                "raw_responses": evaluation_data
            }
    
    def save_results(self, results: Dict[str, Any], output_file: str = "ragas_evaluation_results_fixed.json"):
        """Save evaluation results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Results saved to: {output_file}")
            
            # Also create a summary file
            if "overall_metrics" in results:
                summary_file = "ragas_evaluation_summary_fixed.txt"
                with open(summary_file, 'w') as f:
                    f.write("🎯 RAGAS EVALUATION SUMMARY\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Evaluation Date: {results['evaluation_timestamp']}\n")
                    f.write(f"Dataset: {results['dataset_info']['dataset_path']}\n")
                    f.write(f"Total Questions: {results['dataset_info']['total_questions']}\n\n")
                    f.write("OVERALL METRICS:\n")
                    f.write("-" * 20 + "\n")
                    for metric, score in results["overall_metrics"].items():
                        f.write(f"{metric.replace('_', ' ').title()}: {score:.3f}\n")
                
                logger.info(f"📋 Summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"💥 Failed to save results: {e}")

async def main():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"AIE7 - LangGraph - {uuid4().hex[0:8]}"
    
    """Main evaluation function."""
    print("🎯 Starting Agent RAGAS Evaluation - FIXED VERSION")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = AgentRAGASEvaluator("dataset.csv")
        
        # Run evaluation
        results = await evaluator.evaluate_dataset()
        
        # Save results
        evaluator.save_results(results)
        
        # Print summary
        if "overall_metrics" in results:
            print("\n📊 EVALUATION SUMMARY:")
            print("-" * 30)
            for metric, score in results["overall_metrics"].items():
                emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.7 else "🔴"
                print(f"{emoji} {metric.replace('_', ' ').title()}: {score:.3f}")
        else:
            print("\n⚠️ Evaluation completed but no metrics available")
            print("Check the JSON file for detailed error information")
        
        print(f"\n✅ Evaluation completed! Check 'ragas_evaluation_results_fixed.json' for full results.")
        
    except Exception as e:
        logger.error(f"💥 Evaluation failed: {e}")
        print(f"❌ Evaluation error: {e}")

if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(main())