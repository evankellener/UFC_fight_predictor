#!/bin/bash
# Monitor the deep genetic search progress

echo "🔍 Monitoring Deep Genetic Algorithm Search"
echo "=============================================="
echo ""

# Check if process is running
if ps -p 12038 > /dev/null 2>&1; then
    echo "✅ Process is RUNNING (PID: 12038)"
else
    echo "❌ Process has STOPPED"
fi

echo ""
echo "📊 Latest Progress:"
echo "-------------------"
tail -30 genetic_deep_search.log

echo ""
echo "💡 Commands:"
echo "   View full log:     tail -f genetic_deep_search.log"
echo "   Check if running:  ps -p 12038"
echo "   Stop process:      kill 12038"

