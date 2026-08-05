# Annotation Time and Progress Plan

Use measured active annotation time, not wall-clock time. For an annotator with `n`
completed items over `h` active hours:

`rate = n / h`, `remaining_hours = (27,393 - n) / rate`.

The CLI reports total and per-scene progress and estimates remaining hours:

```powershell
.\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code\annotation_app\cli.py progress annotator_A --elapsed-hours 8.5
```

Projected completion date is the current date plus `remaining_hours / planned_daily_active_hours`.
Keep per-scene counts visible to detect queue or resume errors, but never expose class
counts. Suggested checkpoints are the 500-item protocol pilot, then 10%, 25%, 50%,
75%, and 100% of each independent queue. Back up at least daily and before application
or protocol upgrades.

At 120, 180, and 240 annotations/hour, one 27,393-item pass requires approximately
228.3, 152.2, and 114.1 active hours, respectively. These are planning scenarios, not
measured productivity. The real projection must use the annotator's completed speed.
