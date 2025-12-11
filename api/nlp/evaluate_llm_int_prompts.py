#!/usr/bin/env python3
"""
Main Evaluation Script for LLM INT Prompts

Orchestrates the evaluation pipeline:
1. Load notes and expected annotations
2. Adapt prompts and load into model_runner
3. Build/load FAISS indexes for fewshot examples
4. Run LLM inference on each note-prompt combination
5. Evaluate outputs against expected annotations
6. Generate detailed evaluation reports
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import time
from datetime import timedelta

# Import our modules
from prompt_adapter import adapt_int_prompts
from fewshot_builder import FewshotBuilder
from evaluation_engine import evaluate_annotation, batch_evaluate
from model_runner import init_model, run_model_with_prompt, get_prompt

# Import model_runner module to access _PROMPTS
import model_runner as mr


def load_adapted_prompts(prompts_json_path: str | Path):
    """
    Adapt INT prompts and load them into model_runner's global _PROMPTS.
    
    Args:
        prompts_json_path: Path to FBK_scripts/prompts.json
    """
    print("[INFO] Adapting INT prompts for model_runner...")
    adapted_prompts = adapt_int_prompts(prompts_json_path)
    
    # Load into model_runner's global _PROMPTS
    mr._PROMPTS = adapted_prompts
    print(f"[INFO] Loaded {len(adapted_prompts)} adapted prompts into model_runner")


def main(
    notes_csv_path: str | Path = None,
    mapping_csv_path: str | Path = None,
    json_file_path: str | Path = None,
    prompts_json_path: str | Path = None,
    model_path: str | Path = None,
    faiss_store_dir: str | Path = "faiss_store",
    fewshot_k: int = 5,
    use_fewshots: bool = True,
    force_rebuild_faiss: bool = False
):
    """
    Main evaluation pipeline.
    
    Args:
        notes_csv_path: Path to first_patient_notes.csv
        mapping_csv_path: Path to first_patient_notes_annotation_mapping.csv
        json_file_path: Path to annotated_patient_notes.json
        prompts_json_path: Path to FBK_scripts/prompts.json
        model_path: Path to LLM model file
        faiss_store_dir: Directory for FAISS indexes
        fewshot_k: Number of fewshot examples to retrieve (ignored if use_fewshots=False)
        use_fewshots: If False, run without few-shot examples (zero-shot mode)
        force_rebuild_faiss: Force rebuild FAISS indexes even if they exist
    """
    script_dir = Path(__file__).resolve().parent
    
    # Set default paths
    if notes_csv_path is None:
        notes_csv_path = script_dir / "first_patient_notes.csv"
    if mapping_csv_path is None:
        mapping_csv_path = script_dir / "first_patient_notes_annotation_mapping.csv"
    if json_file_path is None:
        json_file_path = script_dir / "annotated_patient_notes.json"
    if prompts_json_path is None:
        prompts_json_path = script_dir / "FBK_scripts" / "prompts.json"
    if model_path is None:
        model_path = script_dir / "meta-llama-3.1-8b-instruct-q4_k_m.gguf"
    
    notes_csv_path = Path(notes_csv_path)
    mapping_csv_path = Path(mapping_csv_path)
    json_file_path = Path(json_file_path)
    prompts_json_path = Path(prompts_json_path)
    model_path = Path(model_path)
    
    print("=" * 80)
    print("LLM Evaluation Pipeline for INT Prompts")
    print("=" * 80)
    if use_fewshots:
        print(f"Mode: Few-shot (k={fewshot_k})")
    else:
        print("Mode: Zero-shot (no few-shot examples)")
    print("=" * 80)
    
    # Start overall timer
    overall_start_time = time.time()
    
    # Step 1: Load data
    step_start = time.time()
    print("\n[STEP 1] Loading data...")
    print(f"  Loading notes from: {notes_csv_path}")
    notes_df = pd.read_csv(notes_csv_path, delimiter=';', encoding='utf-8')
    print(f"  Loaded {len(notes_df)} notes")
    
    print(f"  Loading expected annotations from: {mapping_csv_path}")
    mapping_df = pd.read_csv(mapping_csv_path, delimiter=';', encoding='utf-8')
    print(f"  Loaded {len(mapping_df)} note-prompt mappings")
    step_duration = time.time() - step_start
    print(f"  [Time: {step_duration:.2f}s]")
    
    # Step 2: Adapt and load prompts
    step_start = time.time()
    print("\n[STEP 2] Adapting prompts...")
    load_adapted_prompts(prompts_json_path)
    step_duration = time.time() - step_start
    print(f"  [Time: {step_duration:.2f}s]")
    
    # Step 3: Initialize FAISS builder (with GPU for embeddings) - skip if not using fewshots
    builder = None
    if use_fewshots:
        step_start = time.time()
        print("\n[STEP 3] Setting up FAISS indexes for few-shot examples...")
        builder = FewshotBuilder(store_dir=faiss_store_dir, use_gpu=True)
        
        # Get prompt types for preloading
        prompt_types = list(mr._PROMPTS.keys())
        
        # Build indexes if needed
        if force_rebuild_faiss or not (Path(faiss_store_dir) / "gender-int.index").exists():
            print("  Building FAISS indexes from patients 2-3...")
            builder.build_all_int_prompts(
                json_file_path,
                prompts_json_path,
                patient_indices=[1, 2],
                force_rebuild=force_rebuild_faiss
            )
        else:
            print("  FAISS indexes already exist, preloading into memory...")
            # Preload all indexes for faster retrieval
            builder.preload_all_indexes(prompt_types)
        step_duration = time.time() - step_start
        print(f"  [Time: {step_duration:.2f}s]")
    else:
        print("\n[STEP 3] Skipping FAISS setup (running in zero-shot mode without few-shot examples)")
        prompt_types = list(mr._PROMPTS.keys())
    
    # Step 4: Initialize model (ensure GPU usage with larger context)
    step_start = time.time()
    print("\n[STEP 4] Initializing LLM model...")
    print(f"  Model path: {model_path}")
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)
    
    # Force GPU usage by setting environment variable if not set
    import os
    if 'LLAMA_N_GPU_LAYERS' not in os.environ:
        os.environ['LLAMA_N_GPU_LAYERS'] = '-1'  # Use all layers on GPU
        print("  [INFO] Setting LLAMA_N_GPU_LAYERS=-1 for maximum GPU usage")
    
    # Initialize with larger context window to use more VRAM
    # Note: llama.cpp doesn't support true batch inference (multiple prompts in parallel)
    # Each prompt is processed sequentially. To maximize VRAM usage:
    # - Increased n_ctx from 4096 to 8192 (larger context window = more VRAM)
    # - Increased n_batch from 512 to 1024 (more efficient prompt processing)
    # - Added n_ubatch=512 and flash_attn=True for better memory efficiency
    print("  [INFO] Using optimized settings: n_ctx=8192, n_batch=1024, flash_attn=True")
    print("  [INFO] Note: Processing is sequential (one prompt at a time), not batched")
    init_model(str(model_path), n_ctx=8192)  # Increased from 4096 to 8192
    print("  Model initialized successfully")
    
    # Verify GPU usage (check after model load)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"  GPU memory usage: {mem_info.used / (1024**3):.2f}GB / {mem_info.total / (1024**3):.2f}GB")
        print(f"  GPU utilization: {util.gpu}% (compute), {util.memory}% (memory)")
        
        # Check for active processes
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if procs:
            print(f"  Active GPU processes: {len(procs)}")
            for proc in procs:
                print(f"    PID {proc.pid}: {proc.usedGpuMemory / (1024**3):.2f}GB")
        else:
            print(f"  Note: llama.cpp may use GPU but not show as process (CUDA context)")
            print(f"  GPU memory will be used during inference")
        pynvml.nvmlShutdown()
    except ImportError:
        print("  [INFO] pynvml not available for detailed GPU monitoring")
    except Exception as e:
        print(f"  [WARN] Could not check GPU status: {e}")
    
    step_duration = time.time() - step_start
    print(f"  [Time: {step_duration:.2f}s]")
    
    # Step 5: Run LLM inference and evaluation
    step_start = time.time()
    print("\n[STEP 5] Running LLM inference and evaluation...")
    print("  This may take a while...")
    
    results = []
    total_combinations = len(notes_df) * len(prompt_types)
    current = 0
    note_timings = []  # Track timing per note
    
    # Optimize: Reduce max_new_tokens for faster inference (most annotations are short)
    max_new_tokens = 128  # Reduced from 256 for speed
    
    for _, note_row in notes_df.iterrows():
        note_start_time = time.time()
        note_id = note_row['note_id']
        note_text = note_row['text']
        note_date = note_row['date']
        p_id = note_row['p_id']
        report_type = note_row['report_type']
        
        print(f"\n  Processing note: {note_id} ({report_type})")
        
        # Get expected annotations for this note
        note_mappings = mapping_df[mapping_df['note_id'] == note_id]
        
        prompt_timings = []  # Track timing per prompt for this note
        
        for prompt_type in prompt_types:
            current += 1
            prompt_start_time = time.time()
            
            # Get expected annotation (if exists)
            expected_mapping = note_mappings[note_mappings['prompt_type'] == prompt_type]
            expected_annotation = ""
            if not expected_mapping.empty:
                expected_annotation_raw = expected_mapping.iloc[0]['matching_annotation']
                # Convert to string (handle NaN, None, floats, etc.)
                if expected_annotation_raw is None or (isinstance(expected_annotation_raw, float) and pd.isna(expected_annotation_raw)):
                    expected_annotation = ""
                else:
                    expected_annotation = str(expected_annotation_raw)
            
            print(f"    [{current}/{total_combinations}] {prompt_type}", end=" ... ", flush=True)
            
            try:
                # Get fewshot examples (or use empty list if disabled)
                if use_fewshots and builder is not None:
                    fewshot_examples = builder.get_fewshot_examples(
                        prompt_type,
                        note_text,
                        k=fewshot_k
                    )
                else:
                    fewshot_examples = []  # Zero-shot mode
                
                # Build prompt using model_runner's get_prompt
                prompt = get_prompt(
                    task_key=prompt_type,
                    fewshots=fewshot_examples,
                    note_text=note_text
                )
                
                # Print the full prompt
                print(f"\n    Prompt ({len(prompt)} chars):")
                print(f"    {'-' * 70}")
                # Truncate very long prompts for readability, but show substantial portion
                prompt_display = prompt
                if len(prompt_display) > 800:
                    prompt_display = prompt_display[:800] + f"\n    ... [truncated {len(prompt) - 800} more characters] ..."
                # Indent each line for readability
                prompt_lines = prompt_display.split('\n')
                for line in prompt_lines:
                    print(f"    {line}")
                print(f"    {'-' * 70}")
                
                # Run LLM (optimized for speed)
                output = run_model_with_prompt(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,  # Reduced for speed
                    temperature=0.3
                )
                
                llm_output = output["normalized"]
                raw_output = output["raw"]
                
                # Clean annotation: remove "Annotation: " prefix if present
                import re
                # Ensure llm_output is a string
                if llm_output is None:
                    llm_output = ""
                elif not isinstance(llm_output, str):
                    llm_output = str(llm_output)
                
                if llm_output:
                    llm_output = re.sub(
                        r'^\s*annotation\s*:\s*', '', llm_output, flags=re.IGNORECASE).strip()
                
                # Evaluate
                evaluation = evaluate_annotation(
                    expected=expected_annotation,
                    predicted=llm_output,
                    note_id=note_id,
                    prompt_type=prompt_type
                )
                
                # Add additional metadata
                evaluation['note_date'] = note_date
                evaluation['p_id'] = p_id
                evaluation['report_type'] = report_type
                evaluation['raw_output'] = raw_output
                evaluation['fewshots_used'] = len(fewshot_examples)
                evaluation['note_text'] = note_text  # Include full note text for comparison
                evaluation['note_text_preview'] = note_text[:200] + '...' if len(note_text) > 200 else note_text  # Preview for CSV
                evaluation['expected_annotation'] = expected_annotation  # Already in evaluation but making explicit
                evaluation['llm_output'] = llm_output  # Explicit LLM output (same as predicted_annotation but clearer naming)
                
                # Add timing
                prompt_duration = time.time() - prompt_start_time
                evaluation['processing_time_seconds'] = round(prompt_duration, 3)
                prompt_timings.append(prompt_duration)
                
                results.append(evaluation)
                
                # Print status with annotations
                match_status = "✓ Match" if evaluation['overall_match'] else "✗ Mismatch"
                print(f"{match_status} (sim: {evaluation['similarity_score']:.2f}, {prompt_duration:.2f}s)")
                
                # Print expected and predicted annotations
                expected_display = expected_annotation if expected_annotation else "[NO EXPECTED ANNOTATION]"
                predicted_display = llm_output if llm_output else "[NO PREDICTION]"
                
                # Truncate long annotations for readability (show first 150 chars)
                max_display_len = 150
                if len(expected_display) > max_display_len:
                    expected_display = expected_display[:max_display_len] + "..."
                if len(predicted_display) > max_display_len:
                    predicted_display = predicted_display[:max_display_len] + "..."
                
                print(f"  Expected:  {expected_display}")
                print(f"  Predicted: {predicted_display}")
                
            except Exception as e:
                prompt_duration = time.time() - prompt_start_time
                print(f"✗ ERROR: {e} ({prompt_duration:.2f}s)")
                # Add error result - ensure all values are strings
                error_expected = str(expected_annotation) if expected_annotation else ""
                results.append({
                    'note_id': str(note_id) if note_id else "",
                    'prompt_type': str(prompt_type) if prompt_type else "",
                    'exact_match': False,
                    'similarity_score': 0.0,
                    'error': str(e),
                    'expected_annotation': error_expected,
                    'predicted_annotation': '',
                    'llm_output': '',
                    'raw_output': '',
                    'overall_match': False,
                    'processing_time_seconds': round(prompt_duration, 3),
                    'note_text': str(note_text) if note_text else "",
                    'note_text_preview': (str(note_text)[:200] + '...') if note_text and len(str(note_text)) > 200 else (str(note_text) if note_text else ""),
                    'note_date': str(note_date) if note_date else "",
                    'p_id': str(p_id) if p_id else "",
                    'report_type': str(report_type) if report_type else "",
                    'fewshots_used': 0
                })
                prompt_timings.append(prompt_duration)
        
        # Record note-level timing
        note_duration = time.time() - note_start_time
        note_timings.append({
            'note_id': note_id,
            'total_time_seconds': round(note_duration, 2),
            'num_prompts': len(prompt_types),
            'avg_time_per_prompt_seconds': round(note_duration / len(prompt_types), 3),
            'prompt_timings': prompt_timings
        })
        print(f"  Note {note_id} completed in {note_duration:.2f}s (avg {note_duration/len(prompt_types):.3f}s per prompt)")
    
    step_duration = time.time() - step_start
    print(f"\n  [STEP 5 completed in {step_duration:.2f}s]")
    
    # Step 6: Generate reports
    step_start = time.time()
    print("\n[STEP 6] Generating evaluation reports...")
    
    results_df = pd.DataFrame(results)
    
    # Detailed CSV with note text and annotations for comparison
    detailed_csv_path = script_dir / "llm_evaluation_detailed.csv"
    # Convert value_details to JSON string for CSV
    results_df_csv = results_df.copy()
    if 'value_details' in results_df_csv.columns:
        results_df_csv['value_details'] = results_df_csv['value_details'].apply(
            lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
        )
    
    # Ensure columns are in a logical order for comparison
    column_order = [
        'note_id', 'p_id', 'note_date', 'report_type', 'prompt_type',
        'note_text_preview', 'note_text',  # Note text (preview + full)
        'expected_annotation', 'predicted_annotation', 'llm_output', 'raw_output',  # Annotations for comparison
        'exact_match', 'similarity_score', 'overall_match',  # Match results
        'total_values', 'values_matched', 'value_match_rate', 'value_details',  # Value-level details
        'processing_time_seconds', 'fewshots_used'  # Metadata
    ]
    # Add any remaining columns not in the order list
    existing_cols = list(results_df_csv.columns)
    for col in existing_cols:
        if col not in column_order:
            column_order.append(col)
    # Reorder columns (only include columns that exist)
    results_df_csv = results_df_csv[[col for col in column_order if col in results_df_csv.columns]]
    
    results_df_csv.to_csv(detailed_csv_path, index=False, encoding='utf-8', sep=';')
    print(f"  Detailed results saved to: {detailed_csv_path}")
    print(f"    Includes note text and annotations for easy comparison")
    
    # Summary CSV (per prompt type)
    summary_rows = []
    for prompt_type in prompt_types:
        prompt_results = results_df[results_df['prompt_type'] == prompt_type]
        if len(prompt_results) > 0:
            summary = batch_evaluate(prompt_results.to_dict('records'))
            summary['prompt_type'] = prompt_type
            summary_rows.append(summary)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = script_dir / "llm_evaluation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"  Summary saved to: {summary_csv_path}")
    
    step_duration = time.time() - step_start
    print(f"  [Time: {step_duration:.2f}s]")
    
    # Calculate timing statistics
    overall_duration = time.time() - overall_start_time
    processing_times = [r.get('processing_time_seconds', 0) for r in results if 'processing_time_seconds' in r]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # JSON report
    overall_stats = batch_evaluate(results)
    json_report = {
        'overall_statistics': overall_stats,
        'timing_statistics': {
            'total_time_seconds': round(overall_duration, 2),
            'total_time_formatted': str(timedelta(seconds=int(overall_duration))),
            'average_time_per_evaluation_seconds': round(avg_processing_time, 3),
            'total_evaluations': len(results),
            'total_notes': len(notes_df),
            'notes_timings': note_timings
        },
        'per_prompt_type': {
            row['prompt_type']: {k: v for k, v in row.items() if k != 'prompt_type'}
            for row in summary_rows
        },
        'total_evaluations': len(results),
        'total_notes': len(notes_df),
        'total_prompt_types': len(prompt_types)
    }
    
    json_report_path = script_dir / "llm_evaluation_report.json"
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    print(f"  JSON report saved to: {json_report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total evaluations: {overall_stats['total']}")
    print(f"Exact matches: {overall_stats['exact_matches']} ({overall_stats['exact_match_rate']*100:.1f}%)")
    print(f"High similarity (≥0.8): {overall_stats['high_similarity_matches']} ({overall_stats['high_similarity_rate']*100:.1f}%)")
    print(f"Overall matches: {overall_stats['overall_matches']} ({overall_stats['overall_match_rate']*100:.1f}%)")
    print(f"Average similarity: {overall_stats['avg_similarity']:.3f}")
    if overall_stats['avg_value_match_rate'] is not None:
        print(f"Average value match rate: {overall_stats['avg_value_match_rate']*100:.1f}%")
    print("\n" + "-" * 80)
    print("TIMING SUMMARY")
    print("-" * 80)
    print(f"Total execution time: {timedelta(seconds=int(overall_duration))} ({overall_duration:.2f} seconds)")
    print(f"Average time per evaluation: {avg_processing_time:.3f} seconds")
    if note_timings:
        avg_note_time = sum(nt['total_time_seconds'] for nt in note_timings) / len(note_timings)
        print(f"Average time per note: {avg_note_time:.2f} seconds")
        print(f"Fastest note: {min(nt['total_time_seconds'] for nt in note_timings):.2f}s")
        print(f"Slowest note: {max(nt['total_time_seconds'] for nt in note_timings):.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate LLM outputs on INT prompts for first patient notes"
    )
    parser.add_argument(
        "--notes-csv",
        type=str,
        help="Path to first_patient_notes.csv"
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        help="Path to first_patient_notes_annotation_mapping.csv"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        help="Path to annotated_patient_notes.json"
    )
    parser.add_argument(
        "--prompts-json",
        type=str,
        help="Path to FBK_scripts/prompts.json"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to LLM model file"
    )
    parser.add_argument(
        "--faiss-dir",
        type=str,
        default="faiss_store",
        help="Directory for FAISS indexes"
    )
    parser.add_argument(
        "--fewshot-k",
        type=int,
        default=5,
        help="Number of fewshot examples to retrieve (ignored if --no-fewshots is used)"
    )
    parser.add_argument(
        "--no-fewshots",
        action="store_true",
        help="Run in zero-shot mode without few-shot examples"
    )
    parser.add_argument(
        "--force-rebuild-faiss",
        action="store_true",
        help="Force rebuild FAISS indexes"
    )
    
    args = parser.parse_args()
    
    main(
        notes_csv_path=args.notes_csv,
        mapping_csv_path=args.mapping_csv,
        json_file_path=args.json_file,
        prompts_json_path=args.prompts_json,
        model_path=args.model_path,
        faiss_store_dir=args.faiss_dir,
        fewshot_k=args.fewshot_k,
        use_fewshots=not args.no_fewshots,  # Invert: --no-fewshots means use_fewshots=False
        force_rebuild_faiss=args.force_rebuild_faiss
    )

