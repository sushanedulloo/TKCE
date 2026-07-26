# TKCE Diagnosis — a plain-language guide

**What this document is:** an explanation of the `run_diagnosis.py` tool — what it
does, *why* it works the way it does, and how to read what it prints. Written to
be understandable without any prior context. Everything is explained with tiny
examples first, real code second.

---

## 1. The problem we are trying to understand

Our whole idea (the "TKCE" method) is a **relay race** with four runners. A secret
message is passed from one runner to the next, and the last runner uses it to make
a guess:

```
   Tree   ─▶   Kernel   ─▶   Siamese encoder   ─▶   Head   ─▶   final guess
  (knows        (writes         (tries to             (uses
   the truth)    it down)        memorise it)          the memory)
```

The **secret message** is: *"which data points are similar to which other points."*
That message secretly carries the answer, because similar points usually have the
same label.

Here is the trouble. On our hardest, most important dataset (`eye_movements`), when
we run the whole relay and look at the final guess, the scores look like this
(higher AUC = better; AUC is just "how good is the guess", 0.5 = coin flip, 1.0 =
perfect):

| Who is guessing | Score (AUC) |
|---|---|
| A plain tree model (lightgbm) — **the target we want to beat/match** | **0.703** |
| A cheap trick (leaf-one-hot) we are *supposed* to beat | 0.668 |
| **Our method (TKCE joint)** | **0.664** |
| A plain neural net, no tree help at all | 0.640 |

Our fancy method (0.664) **loses to the tree** and only **ties the cheap trick**.
Something is wrong. But that single final number does not tell us **where** it went
wrong. The message got garbled somewhere in the relay — but which runner dropped it?

**That is exactly what my professor asked for:** don't just say "it doesn't work" —
open up the relay and check the message *after each runner* to find the exact spot
where it gets lost. He literally named the steps: *"starting from the kernel, see if
there is any structure that got it. after training the siamese, did it preserve?"*

`run_diagnosis.py` is the tool that does that checking.

---

## 2. The one big idea: check the message at every station

Think of a **doctor**. You feel sick (bad final score). A good doctor does not just
say "you're sick, goodbye." They run a **blood test, an X-ray, a scan** — one test
per organ — until they find the *one* organ that's the problem.

Our pipeline has 4 "organs". So we run one test per organ and see the message get
weaker (or not) as it passes through:

| Station (organ) | The question we ask | My professor's words |
|---|---|---|
| **1. The kernel** | Does the kernel even *contain* the answer? | *"is there any structure that got it?"* |
| **2. The Siamese** | Did the encoder *keep* the kernel's message? | *"after training the siamese, did it preserve?"* |
| **3. The embedding** | Is the encoder's output *still useful* for the label? | (the natural next question) |
| **4. The head** | Does the final runner *actually use* what it got? | (the natural next question) |

The genius part: if the message is strong at station 1 but weak at station 2, then
**station 2 (the Siamese) is the leak.** No more guessing. We *measure* it.

---

## 3. Each checkpoint, with a tiny example

Below, every check is explained first with a made-up mini-example of just a few
data points, then what the tool actually computes.

### Checkpoint 1 — Does the kernel hold the answer?

The kernel is just a big **similarity table**. For every pair of points it stores a
number from 0 to 1: *how similar the trees think they are.*

Imagine 4 patients. Patients 1 & 2 are both "sick" (label 1); patients 3 & 4 are
both "healthy" (label 0). A **good** similarity table would look like this:

|   | P1 | P2 | P3 | P4 |
|---|----|----|----|----|
|P1 | 1.0| **0.9**| 0.1| 0.2|
|P2 |0.9| 1.0| 0.2| 0.1|
|P3 |0.1| 0.2| 1.0| **0.8**|
|P4 |0.2| 0.1| 0.8| 1.0|

See the pattern? Same-label pairs (P1–P2 = 0.9, P3–P4 = 0.8) are **high**, and
different-label pairs (P1–P3 = 0.1) are **low**. That means the table *knows* the
answer. If instead every cell were around 0.5, the table would be useless — pure fog.

**How the tool measures this (three ways):**

1. **Within-class vs between-class similarity.** Average the "same-label" cells
   (0.85 here) and the "different-label" cells (0.15 here). A big gap = good structure.
   The tool prints `within-class K=... vs between-class K=... gap=...`.
2. **Alignment score** — one number from 0 to 1 saying "how much does this table
   look like the perfect same-label/different-label pattern?" (Explained more below.)
3. **The real test: kNN-on-the-kernel.** Take a test patient, find its *k* most
   similar training patients *using only the kernel table*, and let them vote on the
   label. If this voting is accurate, the kernel truly holds the answer. The tool
   prints `kNN ON KERNEL K: AUC=...`.

> **If this comes out bad:** the raw material is bad. There is no point teaching a
> neural net to copy a kernel that doesn't know anything. (I *expect* this one to be
> fine — the kernel is basically the tree's own opinion — but we must **prove** it,
> not assume it.)

### Checkpoint 2 — Did the Siamese *preserve* the kernel? (the key one)

The Siamese encoder's whole job is to look at the kernel's similarity table and
learn to **reproduce it** using neural embeddings. After training, we can build the
encoder's *own* similarity table (call it **Ĝ**) and lay it next to the kernel's
table **K**. The question: **do they match?**

Tiny example. Kernel says P1 & P2 are 0.9 similar. What does the encoder say?

| Pair | Kernel K says | Encoder Ĝ says | Preserved? |
|---|---|---|---|
| P1–P2 (same label) | 0.9 | 0.85 | ✅ yes |
| P3–P4 (same label) | 0.8 | 0.10 | ❌ **lost it!** |
| P1–P3 (diff label) | 0.1 | 0.12 | ✅ yes |

If lots of rows look like the P3–P4 row, the encoder **failed to preserve** the
message. That would be our leak.

**How the tool measures this:**

- **Correlation / alignment between K and Ĝ.** If Ĝ is a faithful copy of K, these
  are near **1.0**. If the encoder ignored the kernel, they're near **0**. The tool
  prints `alignment(Ghat, K)` and `corr(Ghat_ij, K_ij)`.
- **Neighbour recall@k.** For each point, list its top-*k* nearest neighbours
  *according to the kernel*, and its top-*k* neighbours *according to the encoder*.
  How many are the same?
  - Example: point A's kernel-neighbours = {B, C, D}. Its encoder-neighbours =
    {B, C, X}. Two out of three match → **recall@3 = 0.67**.
  - recall near 1.0 = the encoder kept the neighbourhood structure. Near 0 = it
    scrambled everyone's neighbours.

> **If this comes out bad but Checkpoint 1 was good:** the **Siamese encoder is the
> leak.** The kernel had the message, the encoder threw it away. The fix lives in the
> encoder — its loss function, its size, its learning rate, how long it trained.

### Checkpoint 3 — Is the embedding still *useful* for the label?

Here's a sneaky trap: the encoder could copy the kernel *perfectly* and **still** be
useless. Why? Because "matching the kernel" and "being good for predicting the label"
are not exactly the same thing. So we test the embedding **directly**.

We freeze the encoder and put the **simplest possible classifier** (a "linear probe"
— basically drawing one straight line to separate the classes) on top of its output.
If even that simple classifier does well, the embedding is genuinely useful.

Then we compare, apples to apples, the same simple classifier on:

- the **encoder's embedding** (our method),
- the **raw features** (no tree help),
- the **leaf-one-hot** features (the cheap trick we must beat).

Example outcome:

| Simple classifier put on top of... | Score |
|---|---|
| our embedding φ | 0.62 |
| leaf-one-hot (cheap trick) | 0.67 |
| raw features | 0.60 |

If we see this, it's bad news for the thesis: our fancy embedding (0.62) is **worse
than the cheap trick** (0.67). That tells us the contrastive machinery isn't earning
its keep. The tool prints all three `linear probe on ...` lines, plus a `kNN ON
EMBEDDING phi` line.

### Checkpoint 4 — Does the head actually use it?

Finally, compare the **full trained head** (the last runner) against those simple
probes. If the simple probe on the embedding scores 0.66 but the full fancy head only
scores 0.64, then the head is **wasting** a good embedding — the leak is in the head
or in how we train everything together.

---

## 4. The bonus check: did the embedding "collapse"?

Contrastive encoders have a famous failure: to make their loss look good, they cheat
by mapping **every** point to almost the **same spot**. Imagine being asked to place
1000 cities on a map so distances match a table — and you just pile all 1000 on top
of each other. The "error" looks small, but the map is worthless.

We catch this with three simple health checks on the embedding:

- **Effective rank** — the embedding has, say, 128 dimensions (128 directions it
  *could* spread out in). If it only really uses **3** of them, it collapsed.
  (Think: a globe squashed into a flat pancake, then into a single line.)
- **Dead dimensions** — how many of the 128 directions have basically zero variation
  (everyone has the same value there). Lots of dead dims = collapse.
- **Mean pairwise cosine** — if every pair of points has similarity near 1.0,
  everything is piled together.

> **If the embedding collapsed:** that *one* bug explains the entire failure, and it's
> very common and very fixable (temperature, learning rate, loss choice). This is the
> cheapest possible thing to rule out, which is why we always check it.

---

## 5. The star of the show: the "accuracy ladder"

This is the single picture that points the finger. We line up all the stations as
rungs of a ladder, from "the answer is fully there" at the top down to "our final
result" at the bottom, and read off **where the biggest fall happens**.

A made-up example ladder:

```
Tree (the ceiling we want)      0.70   ┐
kNN on the kernel K             0.70   │  no drop → kernel is FINE
kNN on the embedding φ          0.61   │  ⬅── BIG DROP of 0.09 here!
linear probe on φ               0.62   │
full TKCE head                  0.64   ┘
```

Reading it like a detective: the message was strong (0.70) all the way through the
kernel, then **fell off a cliff (0.70 → 0.61) at the encoder step.** Verdict:

> *"The kernel holds the structure, but the Siamese network loses it. The encoder is
> the leak — that's what we fix next."*

That is a **measured, defensible** conclusion, not a guess. It's exactly the "we need
to really **know** it is not working" my professor asked for. A different run might
put the big drop somewhere else, and then we'd fix a different part — but either way,
**the biggest fall names the culprit.**

The tool computes all the rungs and even prints the biggest drop for you:
```
biggest drop: kNN on K -> kNN on phi  (-0.0900)
^ that transition is where the signal is being lost.
```

---

## 6. What the tool produces

One command (run on the GPU server — training is heavy):

```bash
python run_diagnosis.py --task 361070 --epochs 400 --lr 1e-6 --lam 0.015 --device auto
```

It creates, in `results/diagnosis/`:

- **`diag_eye_movements.png`** — a 3×3 picture with:
  the ladder; the kernel table K as a heatmap (you should *see* colored blocks by
  class); the encoder table Ĝ as a heatmap (do the blocks survive?); a scatter of
  "K vs Ĝ" (points on the diagonal line = perfect copy); the neighbour-recall curve;
  the contrastive-loss curve (did it even train?); the per-dimension spread (collapse
  check); a 2D map of the embedding colored by class (do the classes separate?); and
  a bar chart of the three probes.
- **`diag_eye_movements.json`** — every number, so you can quote exact values.
- **`diag_eye_movements_ladder.csv`** — just the ladder rungs.

---

## 7. A cheat-sheet for reading the result

Match what you see in the ladder to the diagnosis:

| What you see | What it means | What to fix |
|---|---|---|
| **kNN on K** is already low | The kernel itself is weak — no structure to copy | The kernel (more/deeper trees, or GBT kernel is wrong for this data) |
| kNN on K high, **kNN on φ drops a lot** | The Siamese **didn't preserve** the kernel | The encoder (loss, embedding size, learning rate, epochs) |
| kNN on φ fine, **linear probe on φ low** | Embedding kept neighbours but isn't cleanly usable | Embedding geometry / head capacity |
| Probes on φ fine, **full head low** | Good embedding, wasted by the head/training | λ balance, optimisation, head size |
| φ **ties or loses to leaf-one-hot** | The fancy method isn't beating the cheap trick | The core thesis — this is *the* number to beat |
| **Effective rank tiny / cosine ≈ 1** | The embedding collapsed to a blob | temperature, learning rate, loss choice |

---

## 8. Why I built it this exact way (my reasoning)

A few deliberate design choices, in case you're asked:

- **I reused your real training recipe.** The encoder in the diagnosis is trained with
  the *same* settings as `run_deep_joint.py` (same encoder shape, same loss, same
  learning rate). If I diagnosed a *different* encoder, the diagnosis wouldn't be about
  the model that actually produced 0.664. It has to be the real patient.
- **Same classifier across rungs where possible.** kNN-on-K and kNN-on-φ use the
  *exact same voting rule* — only the similarity source differs (kernel vs embedding).
  That way, any drop between them is caused **only** by the encoder, nothing else.
  It's a clean controlled comparison.
- **I test on unseen test data** for the ladder, so we measure real generalisation,
  not memorisation.
- **I always check for collapse**, because it's the cheapest bug to rule out and it
  would silently explain everything.
- **Everything lands in one figure + one JSON**, so after a single run you can walk
  into a meeting and say one measured sentence about where the method breaks.

---

## 9. What to do next (suggested)

1. Run it on `eye_movements` (`--task 361070`) — the dataset where the failure is
   biggest, so the leak is easiest to see.
2. Read the ladder with the cheat-sheet above; find the biggest drop.
3. Re-run on a *weak* tree-advantage dataset as a **control** — the leak should be
   smaller there. If it is, that confirms the diagnosis is real and not just a bug.
4. Bring the biggest-drop sentence + the figure back, and we decide which single
   component to fix first.

*Files: the tool is `run_diagnosis.py`; this guide is `DIAGNOSIS_GUIDE.md`.*
