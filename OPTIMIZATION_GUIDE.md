# ClipSorter Optimization Guide

## Overview

This document explains the performance optimizations implemented to speed up the conversion and processing pipeline, particularly for RAW and other media formats.

## Key Optimizations Implemented

### 1. **Parallel File Conversion** (4 workers default)
- **Before**: Files were processed sequentially
- **After**: Up to 4 files can be converted in parallel using ThreadPoolExecutor
- **Benefit**: ~2-4x speedup for mixed media batches

**Config Key**: `conversion_parallel_workers` (default: 4)

```json
{
  "conversion_parallel_workers": 4
}
```

### 2. **Parallel QC Analysis** (2 workers default)
- **Before**: Each file's QC checks were sequential
- **After**: Up to 2 files can be analyzed in parallel
- **Benefit**: ~1.5-2x speedup for QC phase (limited to 2 to avoid resource contention)

**Config Key**: `qc_parallel_workers` (default: 2)

```json
{
  "qc_parallel_workers": 2
}
```

### 3. **Parallel Classification** (4 workers)
- **Before**: File classification was sequential
- **After**: Up to 4 files classified in parallel
- **Benefit**: ~2-3x speedup for classification phase

### 4. **Optimized RAW Conversion Strategy**
- **RAW conversion is the slowest step** (~30+ seconds per file with full processing)
- **Solution**: Use the `"auto"` or `"preview"` strategy instead of `"raw"`

**Config Keys**:
```json
{
  "raw_conversion_strategy": "auto",
  "raw_conversion_timeout_sec": 20
}
```

#### RAW Strategy Options:

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| `"preview"` | ✅✅ Fast | Good (embedded JPEG) | 🏃 Quick sorting, 90% of cases |
| `"auto"` | ✅✅ Fast | Good (preview fallback) | ⚖️ Best balance (RECOMMENDED) |
| `"raw"` | ❌ Slow | Excellent (full decode) | 🎨 Professional processing only |

### 5. **Improved Progress Bars**
- **Enhanced display**: Shows real-time progress with file counts and percentage
- **Per-phase tracking**: Each pipeline step (conversion, QC, classification) has detailed progress
- **Consistent formatting**: 100-char wide progress bars for better visibility

## Performance Impact

### Typical Performance Gains

**Raw photo file batch (215 .ARW files)**:

| Phase | Before | After | Speedup |
|-------|--------|-------|---------|
| Conversion | ~15-20 min (sequential) | ~3-5 min (parallel) | **4-5x** |
| QC Checks | ~5-10 min | ~3-5 min | **1.5-2x** |
| Classification | ~2-3 min | ~1 min | **2-3x** |
| **Total** | **~25-35 min** | **~8-12 min** | **2-3x** |

## Configuration Tuning

### For Maximum Speed (Mixed Media)
```json
{
  "raw_conversion_strategy": "preview",
  "conversion_parallel_workers": 8,
  "qc_parallel_workers": 4
}
```
- ✅ Fastest processing
- ⚠️ Higher CPU/memory usage
- 📌 Best for large batches or powerful hardware

### Balanced (Recommended Default)
```json
{
  "raw_conversion_strategy": "auto",
  "conversion_parallel_workers": 4,
  "qc_parallel_workers": 2
}
```
- ✅ Good speed + stable resource usage
- 📌 Works well on most systems
- ✅ Includes RAW fallback for quality

### Conservative (Limited Resources)
```json
{
  "raw_conversion_strategy": "auto",
  "conversion_parallel_workers": 2,
  "qc_parallel_workers": 1
}
```
- ✅ Lower memory/CPU impact
- ⚠️ Slower processing
- 📌 For laptops or old hardware

## RAW Conversion Details

### Why RAW is Slow

1. **Full RAW decoding** using `rawpy` library takes 20-40 seconds per file
2. **Color science computation** for proper white balance and color space conversion
3. **I/O overhead** reading large uncompressed data from disk

### Preview Extraction (Recommended)

Most RAW files contain an embedded JPEG preview taken by the camera:

- ✅ Extracts in **<1 second** per file
- ✅ Quality is **identical to camera preview**
- ✅ Perfect for sorting/QC workflows
- ✅ Fallback available if embedded preview is missing

### When to Use Full RAW Processing

- 🎨 Professional photo editing (final output)
- 🔧 Archival processing with special color science needs
- 📊 Technical analysis

For sorting/QC workflows, **preview extraction is almost always sufficient**.

## Implementation Details

### Technology Stack

- **Parallelization**: Python `concurrent.futures.ThreadPoolExecutor`
- **Progress Display**: `tqdm` library with dynamic updates
- **RAW Processing**: `rawpy` library (optional, with Pillow fallback)

### Code Architecture

```python
# New parallel QC example
with ThreadPoolExecutor(max_workers=qc_workers) as executor:
    futures = {executor.submit(_run_qc_check, record, config): record 
               for record in converted_records}
    with tqdm(total=len(converted_records), ...) as pbar:
        for future in as_completed(futures):
            path, result = future.result()
            qc_results[path] = result
            pbar.update(1)
```

## Monitoring Performance

### Enable Verbose Output

```powershell
python sort.py "E:\your\folder" --verbose
```

This shows:
- Detailed logging for each file
- RAW conversion methods attempted
- Any processing errors or fallbacks

### Check Performance

Time the pipeline:
```powershell
Measure-Command { python sort.py "E:\your\folder" }
```

## Troubleshooting

### Pipeline Too Slow?

1. Check if most files are RAW
   - Change `raw_conversion_strategy` to `"preview"`

2. Increase parallel workers
   - Set `conversion_parallel_workers: 8` or higher
   - Monitor system CPU/memory usage

3. Use `--verbose` flag to find bottlenecks

### Out of Memory?

1. Reduce parallel workers:
   - Set `conversion_parallel_workers: 2`
   - Set `qc_parallel_workers: 1`

2. Use preview strategy for RAW:
   - Set `raw_conversion_strategy: "preview"`

3. Process smaller batches

### RAW Files Failing?

1. Ensure `rawpy` is installed:
   ```powershell
   python -m pip install rawpy
   ```

2. Use `"auto"` strategy (fallback to preview)

3. Check `--verbose` output for specific errors

## Performance Testing

Run the test suite to ensure optimizations don't break functionality:

```powershell
python -m pytest tests/test_sort.py -q
python -m pytest tests/test_qc_photo.py -q
python -m pytest tests/test_converter.py -q
```

## Future Optimization Opportunities

- [ ] GPU acceleration for video transcoding (ffmpeg-nvenc)
- [ ] Batch RAW processing with shared YOLO model
- [ ] Memory-mapped file I/O for large videos
- [ ] Async/await refactoring for I/O-bound operations
- [ ] Custom RAW profile caching for repeated processing

## References

- [rawpy Documentation](https://letmaik.github.io/rawpy/)
- [concurrent.futures Tutorial](https://docs.python.org/3/library/concurrent.futures.html)
- [tqdm Progress Bar](https://tqdm.github.io/)
