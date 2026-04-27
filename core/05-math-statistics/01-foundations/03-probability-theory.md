# Section 4 — Probability Theory

## Lectures covered
- Probability Basics · Quiz
- Addition and Multiplication Rule · Quiz
- Conditional Probability and Bayes Theorem · Quiz
- Chapter Summary

---

## 1. Probability — the basics

A probability is a number in [0, 1] indicating how likely an event is.

$$P(A) = \frac{\text{favorable outcomes}}{\text{total outcomes}}$$

(For equally-likely outcomes; more general definition uses measure theory but you don't need it now.)

### Sample space and events
- **Sample space (Ω)**: all possible outcomes (rolling a die: {1,2,3,4,5,6})
- **Event**: a subset of the sample space ("rolling even" = {2,4,6})

### Properties
- $0 \le P(A) \le 1$
- $P(\Omega) = 1$
- $P(\emptyset) = 0$
- $P(\bar{A}) = 1 - P(A)$ (complement)

### Frequentist vs Bayesian (one paragraph each)
- **Frequentist**: probability = long-run frequency. "P(heads) = 0.5 because in many flips, half are heads."
- **Bayesian**: probability = degree of belief. "I'm 70% sure it'll rain tomorrow." Updated as evidence arrives.

ML uses both. The **Bayes' theorem** ahead is the bridge.

---

## 2. Addition rule

For *any* events A and B:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Subtract the intersection because we'd otherwise double-count it.

For **mutually exclusive** events (can't both happen):
$$P(A \cup B) = P(A) + P(B)$$

### Example
A card drawn from a standard deck.
- P(king) = 4/52
- P(heart) = 13/52
- P(king AND heart) = P(king of hearts) = 1/52
- P(king OR heart) = 4/52 + 13/52 − 1/52 = 16/52

---

## 3. Multiplication rule

For two events A and B:
$$P(A \cap B) = P(A) \cdot P(B \mid A)$$

If A and B are **independent** (one doesn't affect the other):
$$P(A \cap B) = P(A) \cdot P(B)$$

### Example — coin flips (independent)
- P(2 heads in a row) = 0.5 × 0.5 = 0.25

### Example — drawing cards without replacement (dependent)
Draw 2 cards from a deck:
- P(first ace) = 4/52
- P(second ace given first was ace) = 3/51
- P(both aces) = (4/52) × (3/51) ≈ 0.0045

---

## 4. Conditional probability

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

"Probability of A *given* B happened."

### Example — disease test
- 1% of population has disease X
- Test correctly identifies sick people 99% of the time (sensitivity)
- Test incorrectly flags healthy people 5% of the time (false positive rate)

You test positive. **What's the probability you're actually sick?**

Most people guess ~95%. The actual answer is shocking, and Bayes' theorem (next) will give it to us: ~17%.

---

## 5. Bayes' theorem — the engine

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

Where:
- $P(A \mid B)$ — **posterior**: what we want (prob of A given evidence B)
- $P(B \mid A)$ — **likelihood**: prob of evidence given A is true
- $P(A)$ — **prior**: prob of A before any evidence
- $P(B)$ — **evidence**: total prob of B (often expanded via total probability)

### Expanded form (total probability)
$$P(B) = P(B \mid A) \cdot P(A) + P(B \mid \bar{A}) \cdot P(\bar{A})$$

So:
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid \bar{A}) \cdot P(\bar{A})}$$

---

## 6. Bayes — the disease example, worked

Let:
- A = you have disease
- B = you tested positive
- P(A) = 0.01 (prior — disease prevalence)
- P(B|A) = 0.99 (sensitivity)
- P(B|¬A) = 0.05 (false positive rate)

$$P(A \mid B) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99}$$

$$= \frac{0.0099}{0.0099 + 0.0495} = \frac{0.0099}{0.0594} \approx 0.167$$

So **only ~17% chance you're actually sick** despite a positive test. This is because the disease is rare; most positive tests are false positives.

### The lesson — base rates dominate
A test's accuracy (99% / 95%) feels great in isolation. But when the **base rate** (prior) is small, even a great test produces mostly false alarms. This is why screening healthy populations for rare diseases is controversial.

ML models inherit this. A 99%-accurate fraud classifier is often wrong about which transactions are fraudulent — because fraud is rare.

---

## 7. Bayes — Codebasics' typical example

**Spam filter**:
- A = email is spam
- B = email contains the word "lottery"
- P(A) = 0.20 (20% of email is spam)
- P(B|A) = 0.90 (90% of spam contains "lottery")
- P(B|¬A) = 0.05 (5% of legit email mentions "lottery")

$$P(\text{spam} \mid \text{lottery}) = \frac{0.9 \times 0.2}{0.9 \times 0.2 + 0.05 \times 0.8} = \frac{0.18}{0.22} \approx 0.82$$

So an email containing "lottery" is 82% likely to be spam. Multiply across many words (Naive Bayes assumption — independence) and you get the classic spam-filter algorithm.

---

## 8. Independence vs mutual exclusivity (the most-confused pair)

| | Mutually exclusive | Independent |
|---|---|---|
| Definition | Can't happen together: $P(A \cap B) = 0$ | One doesn't affect the other: $P(A \cap B) = P(A) P(B)$ |
| If both have non-zero prob | They are NOT independent | They CAN happen together |
| Example | "It's Monday" and "It's Tuesday" | "It's Monday" and "It's raining" |

Easy to confuse — drill this.

---

## 9. Practical patterns

### Computing P(at least one)
$$P(\text{at least one}) = 1 - P(\text{none})$$

E.g., probability of at least one head in 3 flips:
$$1 - 0.5^3 = 1 - 0.125 = 0.875$$

Always easier than enumerating "1 head OR 2 heads OR 3 heads."

### Combinatorics quick reference
- Permutations (ordered): $n! / (n-k)!$
- Combinations (unordered): $\binom{n}{k} = \frac{n!}{k!(n-k)!}$

```python
from math import comb, perm
comb(52, 5)        # # ways to draw a 5-card poker hand
perm(10, 3)        # # ordered arrangements of 3 from 10
```

---

## 10. Common pitfalls

| Mistake | Cause | Fix |
|---|---|---|
| Adding probabilities of overlapping events | forgot intersection | use $P(A) + P(B) - P(A \cap B)$ |
| Multiplying probabilities of dependent events | assumed independence | check, use conditional prob |
| Reading $P(A|B)$ as $P(B|A)$ | order matters! | "given" is the *condition*, not the focus |
| Ignoring base rates | "test is 99% accurate, must be sick" | always invoke Bayes |
| Confusing mutually exclusive ↔ independent | classic | mutually exclusive = can't co-occur; independent = don't influence |

## Self-check

- [ ] State the addition rule and when it simplifies for mutually exclusive events.
- [ ] State the multiplication rule and when it simplifies for independent events.
- [ ] What's the difference between $P(A|B)$ and $P(B|A)$?
- [ ] State Bayes' theorem and walk through the disease-test example.
- [ ] Why is "99%-accurate test for a rare disease" misleading?
- [ ] Compute: P(at least one 6 in 4 rolls of a fair die).
- [ ] Are "rolling even" and "rolling > 3" independent on a die? Show.
- [ ] What's a "prior" in Bayesian terms?
