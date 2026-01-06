"""Evaluation result analyzer - compare two model evaluations."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

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
        model_name_a = self.stats_a["metadata"]["model_name"]
        model_name_b = self.stats_b["metadata"]["model_name"]
        self.qa_type_a = self.stats_a["metadata"].get("qa_type", "unknown")
        self.qa_type_b = self.stats_b["metadata"].get("qa_type", "unknown")
        
        # Combine model name with qa_type
        self.model_a = f"{model_name_a} ({self.qa_type_a})"
        self.model_b = f"{model_name_b} ({self.qa_type_b})"
        
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
    
    def _get_trace(self, model: str, dataset_id: str) -> Union[str, List]:
        """Get trace for a sample from memory or results.
        
        Args:
            model: 'a' or 'b'
            dataset_id: Dataset idx
            
        Returns:
            - String (response) for direct QA mode
            - List (trace array) for tool mode
        """
        memory_dir = self.memory_dir_a if model == 'a' else self.memory_dir_b
        results_dict = self.results_dict_a if model == 'a' else self.results_dict_b
        qa_type = self.qa_type_a if model == 'a' else self.qa_type_b
        
        # Tool mode: load trace from memory directory
        if qa_type == "tool" and memory_dir:
            memory_path = Path(memory_dir) / "tasks"
            if memory_path.exists():
                for task_folder in memory_path.iterdir():
                    if task_folder.is_dir():
                        trace_file = task_folder / "trace.json"
                        if trace_file.exists():
                            with open(trace_file, 'r', encoding='utf-8') as f:
                                trace_data = json.load(f)
                                if str(trace_data.get("dataset_id")) == str(dataset_id):
                                    return trace_data.get("trace", [])
        
        # Direct mode: use response string from results
        return results_dict[dataset_id].get("response", "")
    
    def _used_tools(self, trace: Union[str, List]) -> bool:
        """Check if tools were used based on trace.
        
        Args:
            trace: Trace data (string or list)
            
        Returns:
            True if tools were used, False otherwise
        """
        if isinstance(trace, list):
            # Tool mode: check if there are action steps in trace
            return any(step.get("type") == "action" for step in trace)
        return False
    
    def analyze(self) -> Dict:
        """Perform analysis and categorize samples.
        
        Returns:
            Dict with categorized samples
        """
        both_correct = []
        both_wrong = []
        only_a_wrong = []
        only_b_wrong = []
        
        for idx in self.results_dict_a.keys():
            ra = self.results_dict_a[idx]
            rb = self.results_dict_b[idx]
            
            # Get traces
            trace_a = self._get_trace('a', idx)
            trace_b = self._get_trace('b', idx)
            
            # Check tool usage
            a_used_tools = self._used_tools(trace_a)
            b_used_tools = self._used_tools(trace_b)
            
            # Build sample record (flat structure)
            sample = {
                "idx": idx,
                "question": ra["question"],
                "ground_truth": ra["ground_truth"],
                "type": ra.get("type", "unknown"),
                "sub_task": ra.get("sub_task", ""),
                "dataset": ra.get("dataset", ""),
                "model_a_predicted": ra["predicted"],
                "model_a_is_correct": ra["is_correct"],
                "model_a_trace": trace_a,
                "model_a_used_tools": a_used_tools,
                "model_b_predicted": rb["predicted"],
                "model_b_is_correct": rb["is_correct"],
                "model_b_trace": trace_b,
                "model_b_used_tools": b_used_tools,
            }
            
            # Categorize
            if ra["is_correct"] and rb["is_correct"]:
                both_correct.append(sample)
            elif not ra["is_correct"] and not rb["is_correct"]:
                both_wrong.append(sample)
            elif not ra["is_correct"] and rb["is_correct"]:
                only_a_wrong.append(sample)
            else:  # ra["is_correct"] and not rb["is_correct"]
                only_b_wrong.append(sample)
        
        return {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "only_a_wrong": only_a_wrong,
            "only_b_wrong": only_b_wrong
        }
    
    def compute_stats(self, analysis: Dict) -> Dict:
        """Compute comparison statistics.
        
        Args:
            analysis: Categorized samples
            
        Returns:
            Statistics dict
        """
        total = sum(len(analysis[key]) for key in analysis)
        
        # Compute tool usage stats for model A
        a_correct_with_tools = sum(1 for s in self.results_a 
                                   if s["is_correct"] and self._used_tools(self._get_trace('a', s["idx"])))
        a_correct_without_tools = self.stats_a["correct"] - a_correct_with_tools
        a_wrong_with_tools = sum(1 for s in self.results_a 
                                if not s["is_correct"] and self._used_tools(self._get_trace('a', s["idx"])))
        a_wrong_without_tools = (self.stats_a["total"] - self.stats_a["correct"]) - a_wrong_with_tools
        
        # Compute tool usage stats for model B
        b_correct_with_tools = sum(1 for s in self.results_b 
                                   if s["is_correct"] and self._used_tools(self._get_trace('b', s["idx"])))
        b_correct_without_tools = self.stats_b["correct"] - b_correct_with_tools
        b_wrong_with_tools = sum(1 for s in self.results_b 
                                if not s["is_correct"] and self._used_tools(self._get_trace('b', s["idx"])))
        b_wrong_without_tools = (self.stats_b["total"] - self.stats_b["correct"]) - b_wrong_with_tools
        
        stats = {
            "models": {
                "model_a": self.model_a,
                "model_b": self.model_b,
            },
            "total": total,
            "model_a": {
                "correct": self.stats_a["correct"],
                "wrong": self.stats_a["total"] - self.stats_a["correct"],
                "accuracy": self.stats_a["accuracy"],
                "correct_with_tools": a_correct_with_tools,
                "correct_without_tools": a_correct_without_tools,
                "wrong_with_tools": a_wrong_with_tools,
                "wrong_without_tools": a_wrong_without_tools,
            },
            "model_b": {
                "correct": self.stats_b["correct"],
                "wrong": self.stats_b["total"] - self.stats_b["correct"],
                "accuracy": self.stats_b["accuracy"],
                "correct_with_tools": b_correct_with_tools,
                "correct_without_tools": b_correct_without_tools,
                "wrong_with_tools": b_wrong_with_tools,
                "wrong_without_tools": b_wrong_without_tools,
            },
            "comparison": {
                "both_correct": {
                    "total": len(analysis["both_correct"]),
                    "a_with_tools": sum(1 for s in analysis["both_correct"] if s["model_a_used_tools"]),
                    "a_without_tools": sum(1 for s in analysis["both_correct"] if not s["model_a_used_tools"]),
                    "b_with_tools": sum(1 for s in analysis["both_correct"] if s["model_b_used_tools"]),
                    "b_without_tools": sum(1 for s in analysis["both_correct"] if not s["model_b_used_tools"]),
                },
                "both_wrong": {
                    "total": len(analysis["both_wrong"]),
                    "a_with_tools": sum(1 for s in analysis["both_wrong"] if s["model_a_used_tools"]),
                    "a_without_tools": sum(1 for s in analysis["both_wrong"] if not s["model_a_used_tools"]),
                    "b_with_tools": sum(1 for s in analysis["both_wrong"] if s["model_b_used_tools"]),
                    "b_without_tools": sum(1 for s in analysis["both_wrong"] if not s["model_b_used_tools"]),
                },
                "only_a_wrong": {
                    "total": len(analysis["only_a_wrong"]),
                    "a_with_tools": sum(1 for s in analysis["only_a_wrong"] if s["model_a_used_tools"]),
                    "a_without_tools": sum(1 for s in analysis["only_a_wrong"] if not s["model_a_used_tools"]),
                    "b_with_tools": sum(1 for s in analysis["only_a_wrong"] if s["model_b_used_tools"]),
                    "b_without_tools": sum(1 for s in analysis["only_a_wrong"] if not s["model_b_used_tools"]),
                },
                "only_b_wrong": {
                    "total": len(analysis["only_b_wrong"]),
                    "a_with_tools": sum(1 for s in analysis["only_b_wrong"] if s["model_a_used_tools"]),
                    "a_without_tools": sum(1 for s in analysis["only_b_wrong"] if not s["model_a_used_tools"]),
                    "b_with_tools": sum(1 for s in analysis["only_b_wrong"] if s["model_b_used_tools"]),
                    "b_without_tools": sum(1 for s in analysis["only_b_wrong"] if not s["model_b_used_tools"]),
                },
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
        logger.info(f"    Correct: {stats['model_a']['correct']} (with tools: {stats['model_a']['correct_with_tools']}, without: {stats['model_a']['correct_without_tools']})")
        logger.info(f"    Wrong: {stats['model_a']['wrong']} (with tools: {stats['model_a']['wrong_with_tools']}, without: {stats['model_a']['wrong_without_tools']})")
        logger.info(f"    Accuracy: {stats['model_a']['accuracy']*100:.2f}%")
        logger.info(f"")
        logger.info(f"  {stats['models']['model_b']}:")
        logger.info(f"    Correct: {stats['model_b']['correct']} (with tools: {stats['model_b']['correct_with_tools']}, without: {stats['model_b']['correct_without_tools']})")
        logger.info(f"    Wrong: {stats['model_b']['wrong']} (with tools: {stats['model_b']['wrong_with_tools']}, without: {stats['model_b']['wrong_without_tools']})")
        logger.info(f"    Accuracy: {stats['model_b']['accuracy']*100:.2f}%")
        logger.info(f"")
        
        delta = stats['comparison']['delta_accuracy']
        delta_sign = "+" if delta > 0 else ""
        logger.info(f"  Delta: {delta_sign}{delta*100:.2f}%\n")
        
        logger.info(f"📈 Head-to-Head Comparison:")
        
        bc = stats['comparison']['both_correct']
        logger.info(f"  ✅ Both Correct: {bc['total']}")
        logger.info(f"     A: with tools: {bc['a_with_tools']}, without: {bc['a_without_tools']}")
        logger.info(f"     B: with tools: {bc['b_with_tools']}, without: {bc['b_without_tools']}")
        
        bw = stats['comparison']['both_wrong']
        logger.info(f"  ❌ Both Wrong: {bw['total']}")
        logger.info(f"     A: with tools: {bw['a_with_tools']}, without: {bw['a_without_tools']}")
        logger.info(f"     B: with tools: {bw['b_with_tools']}, without: {bw['b_without_tools']}")
        
        oaw = stats['comparison']['only_a_wrong']
        logger.info(f"  🔵 Only {stats['models']['model_a']} Wrong: {oaw['total']}")
        logger.info(f"     A: with tools: {oaw['a_with_tools']}, without: {oaw['a_without_tools']}")
        logger.info(f"     B: with tools: {oaw['b_with_tools']}, without: {oaw['b_without_tools']}")
        
        obw = stats['comparison']['only_b_wrong']
        logger.info(f"  🟢 Only {stats['models']['model_b']} Wrong: {obw['total']}")
        logger.info(f"     A: with tools: {obw['a_with_tools']}, without: {obw['a_without_tools']}")
        logger.info(f"     B: with tools: {obw['b_with_tools']}, without: {obw['b_without_tools']}")
        
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
