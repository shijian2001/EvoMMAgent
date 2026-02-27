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


def make_options(choices, format='letter'):
    """Generate option prefixes in different formats.
    
    Args:
        choices: List of choice texts
        format: 'numeric' or 'letter'
    
    Returns:
        Tuple of (prefix1, prefix2, full_options)
        - prefix1: ['A', 'B', 'C', ...] or ['1', '2', '3', ...]
        - prefix2: ['(A)', '(B)', '(C)', ...] or ['(1)', '(2)', '(3)', ...]
        - full_options: ['(A) choice text', '(B) choice text', ...]
    """
    assert format in ['numeric', 'letter']
    if format == 'numeric':
        prefix1 = [str(i + 1) for i in range(len(choices))]
    else:
        prefix1 = [chr(ord("a") + i).upper() for i in range(len(choices))]
    prefix2 = [f"({p})" for p in prefix1]
    return prefix1, prefix2, [f'{p} {c}' for p, c in zip(prefix2, choices)]


def extract_answer(response: str, special_answer_token: str = "\nAnswer:") -> str:
    """Extract content after answer token.
    
    Args:
        response: Agent's complete response
        special_answer_token: Special token marking the final answer
    
    Returns:
        Extracted answer string (first line after token), or empty if token not found
    """
    token = special_answer_token.lstrip('\n')
    if token not in response:
        return ""
    answer = response.split(token)[-1].strip()
    return answer.split('\n')[0].strip()


def _match_to_choice(predicted: str, choices: list, prefixes: list) -> str:
    """Match predicted answer text to a choice.

    Two strategies applied in order:
      1. Prefix identifier with word boundary (regex) — handles A, (A), A., A), (A) xxx, 3, etc.
      2. Choice text matching — exact first, then substring with min-length guard.

    Args:
        predicted: Predicted answer (already stripped)
        choices: List of choice texts
        prefixes: List of prefix strings (e.g. ['A','B','C'] or ['1','2','3'])

    Returns:
        Matched choice text, or None if no match
    """
    pred = predicted.strip().lower()
    if not pred:
        return None

    # Strategy 1: Leading prefix identifier with word boundary
    prefix_map = {p.lower(): i for i, p in enumerate(prefixes)}
    m = re.match(r'^\(?([a-z]|\d+)\)?(?![a-z0-9])', pred)
    if m and m.group(1) in prefix_map:
        return choices[prefix_map[m.group(1)]].strip()

    # Strategy 2a: Exact match on choice text
    for c in choices:
        if pred == c.strip().lower():
            return c.strip()

    # Strategy 2b: Substring match (min length 2 to avoid single-char false positives)
    for c in choices:
        cl = c.strip().lower()
        if len(cl) > 1 and len(pred) > 1 and (cl in pred or pred in cl):
            return c.strip()

    return None


def evaluate_answer(predicted: str, ground_truth: str, task_type: str, 
                   choices: list = None, option_format: str = 'letter') -> bool:
    """Evaluate answer correctness.
    
    Args:
        predicted: Predicted answer (extracted text after answer token)
        ground_truth: Ground truth answer
        task_type: Task type
        choices: List of choices (for multi-choice questions)
        option_format: 'letter' or 'numeric' for option prefixes
    
    Returns:
        True if correct, False otherwise
    """
    if not predicted:
        return False
    
    if task_type == "multi-choice":
        if not choices:
            # Fallback: direct comparison
            return predicted.strip().lower() == ground_truth.strip().lower()
        
        prefix1, _, _ = make_options(choices, format=option_format)
        matched = _match_to_choice(predicted, choices, prefix1)
        if matched:
            return matched.lower() == ground_truth.strip().lower()
        return False
    
    else:
        # Other task types not implemented yet
        raise NotImplementedError(f"Task type '{task_type}' is not implemented yet")


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
        
        # Store whether using tools (for response saving logic)
        self.use_tools = agent_config.get("tool_bank") is not None
        
        # Store special_answer_token for task instructions
        self.special_answer_token = agent_config.get("special_answer_token", "\nAnswer:")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agent pool once for all batches
        if self.verbose:
            logger.info(f"🔧 Initializing {self.max_concurrent} agents...")
        
        self.agents = [
            MultimodalAgent(**self.agent_config)
            for _ in range(self.max_concurrent)
        ]
        
        if self.verbose:
            logger.info(f"✅ Agent pool ready\n")
    
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
    
    def _update_trace_correctness(self, dataset_id: str, is_correct: bool,
                                   ground_truth: str = ""):
        """Update trace.json with is_correct and ground_truth fields.
        
        Args:
            dataset_id: Dataset sample idx
            is_correct: Whether the answer is correct
            ground_truth: Ground truth answer from the dataset
        """
        memory_dir = self.agent_config.get("memory_dir")
        if not memory_dir:
            return
        
        memory_path = Path(memory_dir) / "tasks"
        if not memory_path.exists():
            return
        
        for task_folder in memory_path.iterdir():
            if task_folder.is_dir():
                trace_file = task_folder / "trace.json"
                if trace_file.exists():
                    try:
                        with open(trace_file, 'r', encoding='utf-8') as f:
                            trace_data = json.load(f)
                        
                        if str(trace_data.get("dataset_id")) == str(dataset_id):
                            trace_data["is_correct"] = is_correct
                            if ground_truth:
                                trace_data["ground_truth"] = ground_truth
                            with open(trace_file, 'w', encoding='utf-8') as f:
                                json.dump(trace_data, f, indent=2, ensure_ascii=False)
                            break
                    except Exception as e:
                        logger.debug(f"Failed to update trace for {dataset_id}: {e}")
    
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
            task_instruction = get_task_instruction(task_type, self.use_zh, self.special_answer_token)
            if task_instruction:
                query = f"{query}\n\n{task_instruction}"
            
            # Build absolute image paths directly from sample
            image_paths = [
                str(self.image_dir / img_path) 
                for img_path in sample['images']
            ]
            
            # Run agent with metadata for memory trace
            result = await agent.act(
                query=query,
                images=image_paths,
                verbose=False,
                return_history=False,
                task_metadata={
                    "dataset_id": str(sample.get("idx", "unknown")),
                    "dataset": sample.get("dataset", ""),
                    "sub_task": sample.get("sub_task", ""),
                    "type": sample.get("type", ""),
                }
            )
            
            # Extract response
            response = result if isinstance(result, str) else result.get("response", "")
            
            # Extract and evaluate answer
            predicted = extract_answer(response, self.special_answer_token)
            is_correct = evaluate_answer(
                predicted=predicted,
                ground_truth=sample['answer'],
                task_type=sample['type'],
                choices=sample.get('choices'),
                option_format='letter'
            )
            
            result_dict = {
                "idx": sample['idx'],
                "question": sample['question'],
                "choices": sample.get('choices', []),
                "query": query,
                "ground_truth": sample['answer'],
                "predicted": predicted,
                "is_correct": is_correct,
                "response": "" if self.use_tools else response,
                "type": sample.get('type', 'unknown'),
                "sub_task": sample.get('sub_task', ''),
                "dataset": sample.get('dataset', ''),
            }
            
            if self.use_tools and self.agent_config.get("memory_dir"):
                self._update_trace_correctness(
                    sample['idx'], is_correct,
                    ground_truth=sample.get('answer', ''),
                )
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Task {sample['idx']} failed: {str(e)}")
            return {
                "idx": sample['idx'],
                "question": sample['question'],
                "choices": sample.get('choices', []),
                "query": "",
                "ground_truth": sample['answer'],
                "predicted": "",
                "is_correct": False,
                "response": "",
                "type": sample.get('type', 'unknown'),
                "sub_task": sample.get('sub_task', ''),
                "dataset": sample.get('dataset', ''),
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
        
        completed = 0
        total = len(samples)

        # Agent queue: each task borrows an agent and returns it when done.
        # This guarantees no two concurrent tasks share the same agent instance,
        # avoiding race conditions on stateful components (e.g. search_experiences_tool).
        agent_queue: asyncio.Queue[MultimodalAgent] = asyncio.Queue()
        for a in self.agents:
            agent_queue.put_nowait(a)

        async def run_with_agent(sample: Dict):
            nonlocal completed
            agent = await agent_queue.get()
            try:
                result = await self.run_single_task(sample, agent)
                completed += 1
                if self.verbose:
                    status = "✅" if result["is_correct"] else "❌"
                    logger.info(f"{status} [{completed}/{total}] Sample {result['idx']}: {result['predicted']}")
                return result
            finally:
                agent_queue.put_nowait(agent)

        results = await asyncio.gather(*[
            run_with_agent(sample) for sample in samples
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
        
        # Add metadata for analyzer (always safe, even if memory_dir is None)
        stats["metadata"] = {
            "dataset_path": str(self.jsonl_path),
            "image_dir": str(self.image_dir),
            "model_name": self.agent_config.get("model_name", "unknown"),
            "memory_dir": self.agent_config.get("memory_dir"),
            "use_tools": self.use_tools,
            "qa_type": "tool" if self.use_tools else "direct",
        }
        
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
        
        # Prepare output paths
        results_path = self.output_dir / "results.jsonl"
        
        # Clear previous results (start fresh)
        if results_path.exists():
            results_path.unlink()
        
        # Process in batches with incremental saving
        all_results = []
        for batch_idx in range(0, len(dataset), self.batch_size):
            batch = dataset[batch_idx:batch_idx + self.batch_size]
            batch_id = batch_idx // self.batch_size + 1
            
            results = await self.run_batch(batch, batch_id)
            all_results.extend(results)
            
            # Incremental save: append batch results to JSONL
            with open(results_path, 'a', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
            if self.verbose:
                logger.info(f"💾 Saved batch {batch_id} results to {results_path}")
        
        # Sort by idx (read back from file for consistency)
        all_results.sort(key=lambda x: x['idx'])
        
        # Compute statistics
        stats = self.compute_stats(all_results)
        
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
        
        # Release resources held by agents (e.g. Reranker HTTP clients)
        for agent in self.agents:
            if hasattr(agent, "close"):
                await agent.close()
        
        return stats
