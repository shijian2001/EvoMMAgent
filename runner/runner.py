"""Runner for processing multiple tasks concurrently with memory system."""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from agent.mm_agent import MultimodalAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')


class Runner:
    """Runner for processing multiple tasks concurrently."""
    
    def __init__(
        self,
        agent_config: Dict[str, Any],
        max_concurrent: int = 10,
    ):
        """Initialize runner.
        
        Args:
            agent_config: Configuration for MultimodalAgent (must include memory config)
            max_concurrent: Maximum concurrent tasks
        """
        self.agent_config = agent_config
        self.max_concurrent = max_concurrent
    
    async def run_single_task(
        self,
        task_data: Dict[str, Any],
        agent: MultimodalAgent,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run a single task.
        
        Args:
            task_data: Task configuration with keys:
                - query: str
                - images: Optional[List[str]]
                - videos: Optional[List[str]]
            agent: MultimodalAgent instance
            verbose: Whether to print execution steps
            
        Returns:
            Result dict with task_id, success, response
        """
        try:
            result = await agent.act(
                query=task_data["query"],
                images=task_data.get("images"),
                videos=task_data.get("videos"),
                verbose=verbose,
                return_history=True
            )
            
            return {
                "task_id": result.get("task_id"),
                "success": result.get("success", False),
                "response": result.get("response", ""),
                "task_data": task_data
            }
        except Exception as e:
            logging.error(f"Task failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "task_id": None,
                "success": False,
                "response": str(e),
                "task_data": task_data
            }
    
    async def run_batch(
        self,
        tasks: List[Dict[str, Any]],
        verbose: bool = True,
        verbose_agent: bool = False
    ) -> List[Dict[str, Any]]:
        """Run a batch of tasks with concurrency control.
        
        Args:
            tasks: List of task configurations
            verbose: Whether to print progress
            verbose_agent: Whether to print agent execution steps
            
        Returns:
            List of results
        """
        if verbose:
            logging.info(f"\n{'='*80}")
            logging.info(f"Starting batch processing: {len(tasks)} tasks")
            logging.info(f"Max concurrent: {self.max_concurrent}")
            logging.info(f"{'='*80}\n")
        
        # Create agent for each concurrent task
        agents = [
            MultimodalAgent(**self.agent_config)
            for _ in range(min(self.max_concurrent, len(tasks)))
        ]
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Track completion
        completed = 0
        total = len(tasks)
        
        async def run_with_semaphore(task_data: Dict, agent: MultimodalAgent):
            nonlocal completed
            async with semaphore:
                result = await self.run_single_task(task_data, agent, verbose=verbose_agent)
                completed += 1
                if verbose:
                    status = "✅" if result["success"] else "❌"
                    task_id = result.get("task_id", "N/A")
                    query_preview = task_data['query'][:60] + "..." if len(task_data['query']) > 60 else task_data['query']
                    logging.info(f"{status} [{completed}/{total}] Task {task_id}: {query_preview}")
                return result
        
        # Run all tasks
        agent_cycle = iter(agents)
        results = await asyncio.gather(*[
            run_with_semaphore(task, next(agent_cycle, agents[0]))
            for task in tasks
        ])
        
        if verbose:
            success_count = sum(1 for r in results if r["success"])
            logging.info(f"\n{'='*80}")
            logging.info(f"Batch completed: {success_count}/{len(tasks)} successful")
            logging.info(f"{'='*80}\n")
        
        return results
    
    async def run_from_dataset(
        self,
        dataset_path: str,
        batch_size: int = 100,
        verbose: bool = True
    ):
        """Run tasks from a dataset file in batches.
        
        Args:
            dataset_path: Path to dataset JSON file
            batch_size: Number of tasks per batch
            verbose: Whether to print progress
        """
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        total_tasks = len(dataset)
        logging.info(f"Loaded {total_tasks} tasks from {dataset_path}")
        
        all_results = []
        
        # Process in batches
        for batch_idx in range(0, total_tasks, batch_size):
            batch = dataset[batch_idx:batch_idx + batch_size]
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing batch {batch_idx//batch_size + 1}")
            logging.info(f"Tasks: {batch_idx + 1} - {batch_idx + len(batch)}")
            logging.info(f"{'='*80}")
            
            results = await self.run_batch(batch, verbose=verbose)
            all_results.extend(results)
        
        return all_results
