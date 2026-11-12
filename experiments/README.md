# Experiment log

Production logic lives in `src/helioswarm/`. This directory holds only the experiment log and
exploratory notebooks (PDF §4, §14.4).

Each scientific experiment gets one entry with the following fields:

| Field | Content |
|---|---|
| Date | ISO date |
| Question | The physical/engineering question being asked |
| Config | Path to the YAML config(s) and their `config_hash` (`helioswarm validate-config`) |
| Code commit | `git rev-parse HEAD` and the dirty flag |
| Result location | Run directory, e.g. under `$HELIOSWARM_RUNS_DIR` |
| Interpretation | What the result shows, including negative or failed results (PDF §13.4) |

## Entries

None yet. Simulation runs become possible at milestone M6.
