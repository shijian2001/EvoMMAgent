"""Evaluation runner for benchmark datasets."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

from agent.mm_agent import MultimodalAgent
from .task_instructions import get_task_instruction

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def extract_answer(response: str, task_type: str) -> str:
    """Extract answer from agent response.
    
    Args:
        response: Agent's complete response
        task_type: Task type (multi-choice / open-ended / yes-no / true-false)
    
    Returns:
        Extracted answer string
    """
    # Method 1: Extract content after "Answer:" token
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
        
        # For multiple choice: extract only the letter
        if task_type == "multi-choice":
            match = re.search(r'^([A-E])', answer)
            if match:
                return match.group(1)
            # Fallback: search anywhere in the answer line
            match = re.search(r'([A-E])', answer.split('\n')[0])
            if match:
                return match.group(1)
        
        # For yes-no: normalize to Yes/No
        elif task_type == "yes-no":
            answer_lower = answer.lower()
            if "yes" in answer_lower:
                return "Yes"
            elif "no" in answer_lower:
                return "No"
        
        # For true-false: normalize to True/False
        elif task_type == "true-false":
            answer_lower = answer.lower()
            if "true" in answer_lower:
                return "True"
            elif "false" in answer_lower:
                return "False"
        
        return answer.split('\n')[0].strip()  # Take first line
    
    # Fallback: Take last non-tool-call line
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not any(tok in line for tok in ["Thought:", "Action:", "Action Input:", "Observation:"]):
            if task_type == "multi-choice":
                match = re.search(r'([A-E])', line)
                if match:
                    return match.group(1)
            elif task_type == "yes-no":
                line_lower = line.lower()
                if "yes" in line_lower:
                    return "Yes"
                elif "no" in line_lower:
                    return "No"
            elif task_type == "true-false":
                line_lower = line.lower()
                if "true" in line_lower:
                    return "True"
                elif "false" in line_lower:
                    return "False"
            return line
    
    return ""


def evaluate_answer(predicted: str, ground_truth: str, task_type: str) -> bool:
    """Evaluate answer correctness.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        task_type: Task type
    
    Returns:
        True if correct, False otherwise
    """
    if not predicted:
        return False
    
    if task_type == "multi-choice":
        # Multiple choice: exact letter match
        pred_letter = re.search(r'([A-E])', predicted)
        gt_letter = re.search(r'([A-E])', ground_truth)
        if pred_letter and gt_letter:
            return pred_letter.group(1) == gt_letter.group(1)
        return predicted.strip().upper() == ground_truth.strip().upper()
    
    elif task_type in ["yes-no", "true-false"]:
        # Yes/No or True/False: case-insensitive match
        return predicted.strip().lower() == ground_truth.strip().lower()
    
    # Open-ended: normalized comparison
    return predicted.strip().lower() == ground_truth.strip().lower()


class Runner:
    """Evaluation runner for benchmark datasets."""
    
    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        agent_config: Dict[str, Any],
        output_dir: str,
        batch_size: int = 32,
        max_concurrent: int = 16,
        verbose: bool = True
    ):
        """Initialize evaluation runner.
        
        Args:
            jsonl_path: Path to JSONL dataset file
            image_dir: Path to image folder
            agent_config: Configuration dict for MultimodalAgent
            output_dir: Output directory for results
            batch_size: Batch size for processing
            max_concurrent: Maximum concurrent tasks per batch
            verbose: Whether to print progress
        """
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.agent_config = agent_config
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        
        # Store language preference for task instructions
        self.use_zh = agent_config.get("use_zh", False)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self) -> List[Dict]:
        """Load dataset from JSONL file.
        
        Returns:
            List of samples
        """
        dataset = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                dataset.append(sample)
        
        if self.verbose:
            logger.info(f"📂 Loaded {len(dataset)} samples from {self.jsonl_path}")
        
        return dataset
    
    async def run_single_task(
        self,
        sample: Dict,
        agent: MultimodalAgent
    ) -> Dict:
        """Run evaluation on a single sample.
        
        Args:
            sample: Dataset sample
            agent: MultimodalAgent instance
        
        Returns:
            Result dict with idx, question, ground_truth, predicted, is_correct
        """
        try:
            # Build query from sample (strip all text fields)
            query = (sample.get('prompt') or sample['question']).strip()
            
            if sample.get('choices') and not sample.get('prompt'):
                # Add choices to query if not in prompt (strip each choice)
                choices_text = '\n'.join([choice.strip() for choice in sample['choices']])
                query = f"{query}\n\n{choices_text}"
            
            # Append task-specific instruction based on task type
            task_type = sample.get('type', 'open-ended')
            task_instruction = get_task_instruction(task_type, self.use_zh)
            if task_instruction:
                query = f"{query}\n\n{task_instruction}"
            
            # Build absolute image paths directly from sample
            image_paths = [
                str(self.image_dir / img_path) 
                for img_path in sample['images']
            ]
            
            # Run agent
            result = await agent.act(
                query=query,
                images=image_paths,
                verbose=False,
                return_history=False
            )
            
            # Extract response
            response = result if isinstance(result, str) else result.get("response", "")
            
            # Extract and evaluate answer
            predicted = extract_answer(response, sample['type'])
            is_correct = evaluate_answer(predicted, sample['answer'], sample['type'])
            
            return {
                "idx": sample['idx'],
                "question": sample['question'],
                "choices": sample.get('choices', []),
                "ground_truth": sample['answer'],
                "predicted": predicted,
                "is_correct": is_correct,
                "response": response  # Keep full response for debugging
            }
            
        except Exception as e:
            logger.error(f"Task {sample['idx']} failed: {str(e)}")
            return {
                "idx": sample['idx'],
                "question": sample['question'],
                "choices": sample.get('choices', []),
                "ground_truth": sample['answer'],
                "predicted": "",
                "is_correct": False,
                "error": str(e)
            }
    
    async def run_batch(
        self,
        samples: List[Dict],
        batch_id: int
    ) -> List[Dict]:
        """Run evaluation on a batch of samples with concurrency control.
        
        Args:
            samples: List of samples
            batch_id: Batch ID for logging
        
        Returns:
            List of results
        """
        if self.verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"Batch {batch_id}: Processing {len(samples)} samples")
            logger.info(f"{'='*80}")
        
        # Create agent pool
        agents = [
            MultimodalAgent(**self.agent_config)
            for _ in range(min(self.max_concurrent, len(samples)))
        ]
        
        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        total = len(samples)
        
        async def run_with_semaphore(sample: Dict, agent: MultimodalAgent):
            nonlocal completed
            async with semaphore:
                result = await self.run_single_task(sample, agent)
                completed += 1
                if self.verbose:
                    status = "✅" if result["is_correct"] else "❌"
                    logger.info(f"{status} [{completed}/{total}] Sample {result['idx']}: {result['predicted']}")
                return result
        
        # Run all tasks with agent cycling
        from itertools import cycle
        agent_cycle = cycle(agents)
        results = await asyncio.gather(*[
            run_with_semaphore(sample, next(agent_cycle))
            for sample in samples
        ])
        
        if self.verbose:
            correct = sum(1 for r in results if r["is_correct"])
            logger.info(f"Batch {batch_id} completed: {correct}/{total} correct ({correct/total*100:.1f}%)")
        
        return results
    
    def compute_stats(self, results: List[Dict]) -> Dict:
        """Compute evaluation statistics.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Statistics dict
        """
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0
        
        stats = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4)
        }
        
        # Load original dataset for grouping
        dataset = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
        
        # Group by type
        by_type = {}
        for sample in dataset:
            task_type = sample['type']
            if task_type not in by_type:
                by_type[task_type] = {"total": 0, "correct": 0}
            by_type[task_type]["total"] += 1
        
        for result in results:
            for sample in dataset:
                if sample['idx'] == result['idx']:
                    task_type = sample['type']
                    if result['is_correct']:
                        by_type[task_type]["correct"] += 1
                    break
        
        # Calculate accuracy by type
        for task_type in by_type:
            t = by_type[task_type]["total"]
            c = by_type[task_type]["correct"]
            by_type[task_type]["accuracy"] = round(c / t, 4) if t > 0 else 0
        
        stats["by_type"] = by_type
        
        # Group by subtask
        by_subtask = {}
        for sample in dataset:
            subtask = sample.get('sub_task', 'unknown')
            if subtask not in by_subtask:
                by_subtask[subtask] = {"total": 0, "correct": 0}
            by_subtask[subtask]["total"] += 1
        
        for result in results:
            for sample in dataset:
                if sample['idx'] == result['idx']:
                    subtask = sample.get('sub_task', 'unknown')
                    if result['is_correct']:
                        by_subtask[subtask]["correct"] += 1
                    break
        
        # Calculate accuracy by subtask
        for subtask in by_subtask:
            t = by_subtask[subtask]["total"]
            c = by_subtask[subtask]["correct"]
            by_subtask[subtask]["accuracy"] = round(c / t, 4) if t > 0 else 0
        
        stats["by_subtask"] = by_subtask
        
        return stats
    
    async def run_evaluation(self) -> Dict:
        """Run full evaluation.
        
        Returns:
            Statistics dict
        """
        if self.verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"🚀 Starting Evaluation")
            logger.info(f"{'='*80}")
            logger.info(f"Dataset: {self.jsonl_path}")
            logger.info(f"Images: {self.image_dir}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Batch size: {self.batch_size}")
            logger.info(f"Max concurrent: {self.max_concurrent}")
            logger.info(f"{'='*80}\n")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Process in batches
        all_results = []
        for batch_idx in range(0, len(dataset), self.batch_size):
            batch = dataset[batch_idx:batch_idx + self.batch_size]
            batch_id = batch_idx // self.batch_size + 1
            
            results = await self.run_batch(batch, batch_id)
            all_results.extend(results)
        
        # Sort by idx
        all_results.sort(key=lambda x: x['idx'])
        
        # Compute statistics
        stats = self.compute_stats(all_results)
        
        # Save results (remove full response for clean output)
        results_clean = [
            {k: v for k, v in r.items() if k != 'response'}
            for r in all_results
        ]
        
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Evaluation Completed")
            logger.info(f"{'='*80}")
            logger.info(f"Results saved to: {results_path}")
            logger.info(f"Statistics saved to: {stats_path}")
            logger.info(f"\n📊 Overall Accuracy: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
            logger.info(f"{'='*80}\n")
        
        return stats
