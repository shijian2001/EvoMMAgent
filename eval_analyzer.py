"""Evaluation result analyzer - compare two model evaluations."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class EvalAnalyzer:
    """Compare two evaluation results."""
    
    def __init__(self, result_dir_a: str, result_dir_b: str):
        """Initialize analyzer.
        
        Args:
            result_dir_a: First evaluation result directory
            result_dir_b: Second evaluation result directory
        """
        self.dir_a = Path(result_dir_a)
        self.dir_b = Path(result_dir_b)
        
        # Load results and stats
        self.results_a = self._load_jsonl(self.dir_a / "results.jsonl")
        self.results_b = self._load_jsonl(self.dir_b / "results.jsonl")
        self.stats_a = self._load_json(self.dir_a / "stats.json")
        self.stats_b = self._load_json(self.dir_b / "stats.json")
        
        # Extract metadata
        self.model_a = self.stats_a["metadata"]["model_name"]
        self.model_b = self.stats_b["metadata"]["model_name"]
        self.memory_dir_a = self.stats_a["metadata"].get("memory_dir")
        self.memory_dir_b = self.stats_b["metadata"].get("memory_dir")
        
        # Validate dataset consistency (efficient: compare idx sets)
        self._validate_consistency()
        
        # Build index for fast lookup
        self.results_dict_a = {r["idx"]: r for r in self.results_a}
        self.results_dict_b = {r["idx"]: r for r in self.results_b}
    
    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _validate_consistency(self):
        """Validate dataset consistency using idx sets."""
        idx_set_a = {r["idx"] for r in self.results_a}
        idx_set_b = {r["idx"] for r in self.results_b}
        
        assert idx_set_a == idx_set_b, \
            f"❌ Dataset mismatch! A has {len(idx_set_a)} samples, B has {len(idx_set_b)} samples"
        
        logger.info(f"✅ Dataset validated: {len(idx_set_a)} samples\n")
    
    def _get_trace(self, model: str, dataset_id: str) -> Dict:
        """Get trace for a sample from memory or results.
        
        Args:
            model: 'a' or 'b'
            dataset_id: Dataset idx
            
        Returns:
            Trace dict (from memory or results response)
        """
        memory_dir = self.memory_dir_a if model == 'a' else self.memory_dir_b
        results_dict = self.results_dict_a if model == 'a' else self.results_dict_b
        
        # If memory enabled, load trace from memory directory
        if memory_dir:
            memory_path = Path(memory_dir) / "tasks"
            if memory_path.exists():
                for task_folder in memory_path.iterdir():
                    if task_folder.is_dir():
                        trace_file = task_folder / "trace.json"
                        if trace_file.exists():
                            with open(trace_file, 'r', encoding='utf-8') as f:
                                trace_data = json.load(f)
                                if str(trace_data.get("dataset_id")) == str(dataset_id):
                                    return trace_data
        
        # Fallback: use response from results (direct QA mode)
        return {"response": results_dict[dataset_id].get("response", "")}
    
    def analyze(self) -> Dict:
        """Perform analysis and categorize samples.
        
        Returns:
            Dict with categorized samples
        """
        both_correct = []
        both_wrong = []
        only_a_correct = []
        only_b_correct = []
        
        for idx in self.results_dict_a.keys():
            ra = self.results_dict_a[idx]
            rb = self.results_dict_b[idx]
            
            # Get traces
            trace_a = self._get_trace('a', idx)
            trace_b = self._get_trace('b', idx)
            
            # Build sample record
            sample = {
                "idx": idx,
                "question": ra["question"],
                "ground_truth": ra["ground_truth"],
                "type": ra.get("type", "unknown"),
                "sub_task": ra.get("sub_task", ""),
                "model_a": {
                    "predicted": ra["predicted"],
                    "is_correct": ra["is_correct"],
                    "trace": trace_a
                },
                "model_b": {
                    "predicted": rb["predicted"],
                    "is_correct": rb["is_correct"],
                    "trace": trace_b
                }
            }
            
            # Categorize
            if ra["is_correct"] and rb["is_correct"]:
                both_correct.append(sample)
            elif not ra["is_correct"] and not rb["is_correct"]:
                both_wrong.append(sample)
            elif ra["is_correct"] and not rb["is_correct"]:
                only_a_correct.append(sample)
            else:
                only_b_correct.append(sample)
        
        return {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "only_a_correct": only_a_correct,
            "only_b_correct": only_b_correct
        }
    
    def compute_stats(self, analysis: Dict) -> Dict:
        """Compute comparison statistics.
        
        Args:
            analysis: Categorized samples
            
        Returns:
            Statistics dict
        """
        total = sum(len(analysis[key]) for key in analysis)
        
        stats = {
            "models": {
                "model_a": self.model_a,
                "model_b": self.model_b
            },
            "total": total,
            "model_a": {
                "correct": self.stats_a["correct"],
                "wrong": self.stats_a["total"] - self.stats_a["correct"],
                "accuracy": self.stats_a["accuracy"]
            },
            "model_b": {
                "correct": self.stats_b["correct"],
                "wrong": self.stats_b["total"] - self.stats_b["correct"],
                "accuracy": self.stats_b["accuracy"]
            },
            "comparison": {
                "both_correct": len(analysis["both_correct"]),
                "both_wrong": len(analysis["both_wrong"]),
                "only_a_correct": len(analysis["only_a_correct"]),
                "only_b_correct": len(analysis["only_b_correct"]),
                "delta_accuracy": self.stats_b["accuracy"] - self.stats_a["accuracy"]
            }
        }
        
        return stats
    
    def print_summary(self, stats: Dict):
        """Print comparison summary."""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 EVALUATION COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"Model A: {stats['models']['model_a']}")
        logger.info(f"Model B: {stats['models']['model_b']}")
        logger.info(f"Total Samples: {stats['total']}")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"🎯 Individual Performance:")
        logger.info(f"  {stats['models']['model_a']}:")
        logger.info(f"    Correct: {stats['model_a']['correct']}")
        logger.info(f"    Wrong: {stats['model_a']['wrong']}")
        logger.info(f"    Accuracy: {stats['model_a']['accuracy']*100:.2f}%")
        logger.info(f"")
        logger.info(f"  {stats['models']['model_b']}:")
        logger.info(f"    Correct: {stats['model_b']['correct']}")
        logger.info(f"    Wrong: {stats['model_b']['wrong']}")
        logger.info(f"    Accuracy: {stats['model_b']['accuracy']*100:.2f}%")
        logger.info(f"")
        
        delta = stats['comparison']['delta_accuracy']
        delta_sign = "+" if delta > 0 else ""
        logger.info(f"  Delta: {delta_sign}{delta*100:.2f}%\n")
        
        logger.info(f"📈 Head-to-Head Comparison:")
        logger.info(f"  ✅ Both Correct: {stats['comparison']['both_correct']}")
        logger.info(f"  ❌ Both Wrong: {stats['comparison']['both_wrong']}")
        logger.info(f"  🔵 Only {stats['models']['model_a']} Correct: {stats['comparison']['only_a_correct']}")
        logger.info(f"  🟢 Only {stats['models']['model_b']} Correct: {stats['comparison']['only_b_correct']}")
        logger.info(f"{'='*80}\n")
    
    def save_analysis(self, analysis: Dict, stats: Dict, output_dir: str):
        """Save analysis results to directory.
        
        Args:
            analysis: Categorized samples
            stats: Comparison statistics
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save stats
        stats_file = output_path / "comparison_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Stats saved: {stats_file}")
        
        # Save categorized JSONL files
        for category, samples in analysis.items():
            jsonl_file = output_path / f"{category}.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"💾 Saved {len(samples)} samples: {jsonl_file}")
        
        logger.info(f"\n✅ All results saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python eval_analyzer.py results/model_a results/model_b -o comparison/
        """
    )
    
    parser.add_argument("result_dir_a", help="First evaluation result directory")
    parser.add_argument("result_dir_b", help="Second evaluation result directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = EvalAnalyzer(args.result_dir_a, args.result_dir_b)
    analysis = analyzer.analyze()
    stats = analyzer.compute_stats(analysis)
    
    # Print summary
    analyzer.print_summary(stats)
    
    # Save results
    analyzer.save_analysis(analysis, stats, args.output)


if __name__ == "__main__":
    main()

