# Stage packet — draft (before writing a kernel)

Diagnose before you plan; plan before you write (guideline 04):
1. Read the previous candidate's artifacts: `isa.json`, `pcs.json`,
   `roofline.json`, `formal_report.json`, `agent_report.md`. If they do not
   exist for the current best, generate them first (`kagent run --stages
   evidence`) — never design from intuition.
2. Name the limiting resource and the number that proves it
   (math/data ratio, top stall reason, efficiency %).
3. Write `docs/draft.md` in the candidate directory: hypothesis, expected
   effect on WHICH metric, risks, and the first two edits you will make.
4. One hypothesis per draft. If you have three ideas, order them by
   (expected gain x probability) and say why.

Anti-patterns that this campaign already paid for:
- designing from intuition (v3's store-coalescing guess was wrong and cost a
  full version);
- optimizing the wrong path (public wrapper vs serving path);
- planning around a kernel you have never profiled.