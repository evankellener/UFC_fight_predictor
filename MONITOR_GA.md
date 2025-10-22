# 🧬 Monitor Genetic Algorithm Progress

## Quick Check
```bash
tail -30 genetic_long_run.log
```

## Live Monitoring (Real-time updates)
```bash
tail -f genetic_long_run.log
```
Press `Ctrl+C` to exit

## Check if Process is Running
```bash
ps -p 13921
```
If running, you'll see the process details. If not, it either finished or crashed.

## See Only Generation Results
```bash
grep "Generation" genetic_long_run.log | tail -10
```

## See Best Results So Far
```bash
grep "NEW BEST" genetic_long_run.log
```

## Full Progress Summary
```bash
echo "Process Status:" && ps -p 13921 && echo "" && echo "Latest:" && tail -30 genetic_long_run.log
```

## When Complete
Look for the file: `genetic_long_results_TIMESTAMP.json`
```bash
ls -lt genetic_long_results_*.json | head -1
```

