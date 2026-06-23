# Why IDs Swap (and the Known Fix)

A readthrough of the re-identification problem we keep hitting — the lollipop
stand stealing a person's ID — and the standard solution from the tracking
literature.

---

## The symptom

A woman walks in front of a fixed display (the lollipop/flag stand). For a
moment her box and the stand's box overlap. She comes out the other side
wearing the **stand's ID** — so her wait timer starts from whenever the *stand*
was first "seen," not when she got in line.

She looks nothing like a lollipop stand. So why does the tracker think she's it?

---

## Why it happens

Our tracker decides "which detection belongs to which existing track" using
**box overlap only** (IoU — how much two boxes cover the same pixels). Appearance
(what someone *looks* like) is only used later, as a fallback, to recover a
track that was already lost.

So at the critical moment:

```
            one detection (the woman, in front of the stand)
                 /                         \
        her own track                  the stand's track
     (box slightly behind)          (box exactly where she is now)
                 \                         /
              IoU picks whichever box overlaps more
                            |
                 the stand's box wins  -> she gets the stand's ID
```

The matcher **never asks "does this detection look like this track?"** at the
moment it matters. The one signal that would obviously save us — she looks like
herself, not a stand — isn't consulted during the match.

It gets worse: once the stand's track grabs her, that track *moves* to follow
her, so it no longer looks "stationary" and our prop-suppression filter stops
hiding it. The stand's ID now rides on her.

---

## The fix the field already settled on

This is a classic, solved problem — the original SORT tracker had exactly this
weakness. The fix, used by every modern tracker (DeepSORT → StrongSORT →
BoT-SORT), is one idea:

> **Put appearance into the primary match, not as an afterthought.**

Instead of matching on box overlap alone, the match score becomes a blend:

```
match cost = (how far apart the boxes are)  +  (how different they look)
```

Now when the woman overlaps the stand:

- **Her own track:** boxes close *and* looks identical -> very low cost -> match.
- **The stand's track:** boxes overlap *but* looks completely different -> high
  cost -> rejected.

She keeps her ID. The stand can't steal it. That's the whole trick — and we
**already compute the "looks like" feature every frame** (the OSNet embedding);
we just don't use it in the primary match yet.

A second, cheaper signal (from OC-SORT) helps too: **direction of motion.** A
person has a consistent heading; a fixed prop has zero velocity. Adding a
"are they moving the same way?" term separates a walking person from a static
object even before appearance.

---

## What we'd change

1. **Appearance-fused matching (the big one).** Turn the match from
   "box overlap only" into "box overlap + looks." Reuses the embeddings we
   already extract. This fixes both the prop-steals-person swap *and* the
   look-alike-people-crossing swap, in one change.
2. **Motion-direction term (optional, cheap).** Adds velocity consistency so a
   mover and a static prop separate on motion alone.
3. **Mask the stand (belt-and-suspenders).** For a *permanent* fixture, the
   surveillance-standard move is to draw an ignore-region over it once, so it's
   never detected as a person in the first place. Surgical, zero risk to real
   people standing elsewhere.

**Recommendation:** build #1. It's the actual industry-standard answer
(StrongSORT / BoT-SORT), and it turns our "SORT with re-id bolted on" into a
proper appearance-aware tracker.

---

## One-line summary

We match people by *where* they are but not *what they look like* — so anything
that overlaps you can take your identity. The fix, used by every modern tracker,
is to match on both at once. We already have the "what they look like" data; we
just need to use it at the right moment.
