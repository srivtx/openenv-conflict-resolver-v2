# teaching a 3B model to plan through a chaotic afternoon — with RL

*OpenEnv Hackathon 2026 | Team Agent (1)*

**▶ demo video (story):** [youtube.com/watch?v=3jPYWWhIKNs](https://www.youtube.com/watch?v=3jPYWWhIKNs)

---

ok so. every assistant on earth ships the same three demos. set a timer. read the weather. remind me to call mom. cool, very productive, very 2014.

now hand one a real afternoon. board review running long over school pickup. visa deadline missing an attachment. insurance autopay just face-planted. hotel cancel window closes in 40. that pile-up is the actual shape of being alive on a wednesday, and every assistant on the market handles it like isolated support tickets — one at a time, no memory of what the last decision did to the rest of the day.

they don't plan. they react. and reacting to an afternoon like that is how you end up paying $300 to cancel a hotel because you "got busy".

so we built an OpenEnv environment that forces planning, and trained a 3B to actually play through it.

## the env in a paragraph

`PersonalAssistantConflictEnv` is a multi-step OpenEnv env with a real mutable world state. agent sees one conflict at a time + the visible calendar. emits a structured JSON action: intent, owner, priority, optional time slot, optional clarification flag, message. env grades it, mutates state, picks what to surface next.

two things make this not just a glorified classifier. clarifications are two-step — you ask now, info comes back next step, then you act on it. and reschedules can spawn cascade conflicts: push the call past 18:00 and suddenly nobody's covering school pickup. congrats, you played yourself.

## procedural episodes (no static fixtures, no cope)

`conflict_generator.generate_episode(seed, difficulty)` builds episodes from parameterized templates. times, owners, urgencies, event names — all sampled per seed. difficulty stacks features:

- **easy**: 3 conflicts, no clarifications, no cascades.
- **medium**: 5 conflicts, ~1 clarification, no cascades.
- **hard**: 7 conflicts, ~2 clarifications, ~1 cascade rule.

splits are disjoint by construction:

- **train**: seeds `1000-1999` (1000 episodes) → SFT data + GRPO rollouts.
- **holdout**: seeds `9000-9099` (100 episodes) → generalization eval.
- **adversarial**: seeds `5000-5009` (10 episodes) → grader probe pool.

`assert_split_disjoint()` runs at module import. if any seed ever leaks across pools, the env literally refuses to load. we are not training on the eval set. (yes, the eval police are watching.)

## world state that actually moves

`WorldState` lives across the entire episode and tracks:

- `calendar`: events with start, end, owner, locked-flag. mutated when the agent reschedules or proposes a plan.
- `pending_clarifications`: queued info reveals waiting to come back.
- `revealed_info`: `conflict_id → revealed text`.
- `cascade_queue`: follow-on conflicts the agent's own actions have spawned.

the conflict queue is dynamic, not a fixed array index. the same conflict can come back (after a clarification reveal). brand-new conflicts can pop up mid-episode (when a cascade rule fires). state actually carries across the episode — which is the whole point of long-horizon, and the thing every "agentic demo" on twitter quietly does not have.

## two-step clarification

when a conflict has a `ClarificationSpec` and the agent picks `ask_clarification`:

1. **step N**: env scores the ask, records the revealed info on the case, keeps the same conflict at the front of the queue.
2. **step N+1**: env re-presents the same conflict with `revealed_info` attached, and grades the post-reveal action — the actual decision.
3. **step N+2**: queue advances to the next conflict.

picking `ask_clarification` on a case with nothing to ask = `clarification_spam` penalty. so much for the "ask everything to be safe" speedrun.

## cascades (the agent makes its own next problem)

every procedural template can carry a `CascadeRule`. on hard difficulty, one of the reschedule cases has a rule that fires when the agent picks a slot past 18:00 with `owner=work`. a brand-new "school pickup uncovered" conflict gets pushed onto the queue and worked through like any other.

put another way: the agent's own decision generates the next problem. without that loop you're doing classification with extra steps and a fancy name.

`test_cascade_appends_followup_conflict` asserts the cascade fires under a perfect-play oracle for `seed=42`. tested invariant. not vibes.

## the grader

- **slot**: regex-strict 24h `HH:MM` parsing. exact = 1.0, ≤30 min off = 0.7, ≤60 min = 0.4, ≤120 min = 0.2, else 0.0. substring shortcuts return 0.0.
- **message**: length + on-topic verb + word diversity. keyword stuffing scores ~0.5; a real on-topic sentence scores 1.0.
- **weights**: `intent 0.40 / owner 0.20 / slot 0.20 / priority 0.10 / clarification 0.05 / message 0.05`. derivation in the module docstring.
- reward range `[0.0, 1.0]`. no floor. no participation trophy.

## anti-hacking penalties (we found the cheese, killed the cheese)

| penalty               | size   | when it fires                                                  |
|-----------------------|--------|-----------------------------------------------------------------|
| `repetitive_intent`   | −0.10  | same intent 3+ steps in a row                                  |
| `premature_finalize`  | −0.15  | `finalize_itinerary` while multiple conflicts still pending     |
| `terminal_early`      | −0.05  | any other terminal-style intent used to short-circuit           |
| `missing_slot`        | −0.05  | `require_slot=True` but no slot supplied                        |
| `clarification_spam`  | −0.05  | asking when the case has no clarification spec                  |
| `short_message`       | −0.04  | message under 16 chars                                          |

every single one of these came from watching the model speedrun a shortcut on early runs and going *"oh, you little gremlin."* patched, retrained, moved on.

## inference

`inference.py` defaults to the trained Qwen 2.5 3B Instruct + LoRA adapter via `transformers` and `peft` when `MODEL_PATH` is set. an HF Router 72B path is available as a baseline. `EVAL_POOL=holdout` runs the holdout pool. `EVAL_POOL=adversarial` runs the probe pool.

if neither token nor model path is set, the script falls back to a heuristic so it still does *something* visible instead of stack-tracing in front of judges. graceful > silent.

## training (the part you came here for)

**model**: Qwen 2.5 3B Instruct, 4-bit quantized through Unsloth. fits a free Colab T4. picked 3B because we wanted to make a small model do a thing the big models can't really do either.

**recipe**: SFT → GRPO. same shape as ChatGPT's post-training, smaller scale, weirder task.

1. **SFT** on procedurally generated train episodes. every example unique (no duplication of three fixtures × 15 — that was the previous version's sin). each clarification case generates *two* SFT examples — the initial ask, and the post-reveal action — so the model actually learns the partial-obs regime instead of guessing around it.
2. **GRPO** on top, using **the env's actual reward** as the GRPO reward. the reward function in the notebook replays the env up to each step, applies the sampled completion as an action, reads `result.reward`. no proxy metrics. no vibes-based reward shaping.

LoRA only, `r=16`. about 1% of the params actually train. base model stays frozen.

**why GRPO**: no critic network (saves VRAM, our entire personality), works out of the box with TRL, and "generate K completions, score them all against the real env, reinforce the best ones" maps cleanly onto a deterministic grader.

## results

![holdout + adversarial scores by training stage](docs/assets/results.png)

| pool                | untrained 3B | after SFT | after SFT + GRPO |
|---------------------|--------------|-----------|------------------|
| holdout (n=40)      | **0.5454**   | **0.9877**| **0.9876**       |
| adversarial (n=10)  | —            | —         | **0.9885**       |

reading these like adults:

- **+0.4423** lift from untrained → SFT on holdout. the model picks up the format and intent routing on episodes it has *never* seen.
- GRPO ≈ SFT here (0.9876 vs 0.9877). on this task, with this much SFT data, SFT is already doing most of the lifting and GRPO holds the score steady without breaking format. honest finding, posting it as-is. not going to hallucinate a delta to feel better about myself.
- **0.9885 on adversarial.** these are episodes specifically built to break the grader's anti-shortcut logic. the model holds.

### grader sanity probes (same run)

deterministic checks on `_slot_score` and `_message_score`, run at the end of the training notebook:

```
slot-score probe (shortcuts -> 0.0; honest matches -> 1.0):
  hint='after 20:30'   slot='later today'              -> 0.00
  hint='after 20:30'   slot='3pm tomorrow'             -> 0.00
  hint='20:30'         slot='reschedule to 20:30'      -> 1.00
  hint='20:30'         slot='21:00'                    -> 0.70   (30 min off, partial credit)
  hint='20:30'         slot='08:00'                    -> 0.00

message-score probe:
  STUFFED : 0.50   ('reschedule, work, urgent.')
  REAL    : 1.00   ('Reschedule the incident review to 20:30 with owner confirmation and follow up note.')
```

substring shortcuts get 0.0. an honest `HH:MM` match gets 1.0 with proportional partial credit when the time is off. an honest message gets 1.0; keyword stuffing caps at 0.5. grader does what it says on the tin.

## kept compatible

things we deliberately did *not* change so existing clients don't break:

- the action schema (6 intents, 6 owners, 4 priorities, slot, needs_clarification, message). backward-compatible.
- the OpenEnv `reset` / `step` / `state` interface and the FastAPI server.
- Docker + HF Space deploy pipeline.
- a tiny set of static fixture tasks for the live UI demo. they exist so the deployed Space stays interactive for visitors. *not* used for training. *not* used for eval.

## what we'd cook with another week

- **curriculum**. ramp difficulty across training instead of starting on hard. we have the knob, we just didn't sweep it.
- **proper expert pass on reward weights**. they're documented and auditable, but they're still our judgment call.
- **bigger procedural template pool**. ~6 templates plus a finalize template. more variety would help generalization.
- **a learned reward model**. the current grader is deterministic, which is great for shortcut shutdown but caps how fuzzy the judgment can get.

## stuff we'd tell our past selves

- if you can't articulate what your environment's *world state* is, it isn't long-horizon. step counter + reward history ≠ state.
- a static fixture is a dataset. train on it AND eval on it, and you've built a leaderboard for your own homework. (we did this once. we are not proud.)
- reward floors hide bugs. don't add them. ever.
- "works on three hand-written examples" and "works on a hundred procedural ones" are different *kinds* of works. one is a demo. the other is a result.

---

## links

- **demo video (story)**: [youtube.com/watch?v=3jPYWWhIKNs](https://www.youtube.com/watch?v=3jPYWWhIKNs)
- **live environment**: [HuggingFace Space](https://huggingface.co/spaces/srivtx/openenv-conflict-resolver-v2)
- **source**: [github.com/srivtx/openenv-conflict-resolver-v2](https://github.com/srivtx/openenv-conflict-resolver-v2)
- **training notebook**: `notebooks/train_grpo_colab.ipynb`

*built with Unsloth and TRL.*
