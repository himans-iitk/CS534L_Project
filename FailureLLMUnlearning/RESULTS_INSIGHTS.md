# Evaluation Results Insights Guide

## Overview
This guide explains what each evaluation metric measures and how to interpret your results to gain insights about LLM unlearning effectiveness.

---

## 📊 Core Metrics Explained

### 1. **verbmem_f** (VerbMem Forget) - **Lower is Better** ✅
**What it measures:** How well the model can reproduce verbatim text from the "forget" set.

**Interpretation:**
- **Lower values (closer to 0%)** = Better unlearning
  - Model cannot recall exact text from forget set
  - Successful unlearning of verbatim memory
- **Higher values (closer to 100%)** = Poor unlearning
  - Model can still reproduce exact text
  - Unlearning failed for verbatim content

**How it works:**
- Uses ROUGE-L score to compare generated text vs. ground truth
- Tests if model can complete prompts with exact text from forget set
- Range: 0-100% (percentage)

**Insight:** This measures **exact memorization** - the most basic form of forgetting.

---

### 2. **privleak** (Privacy Leakage) - **Lower is Better** ✅
**What it measures:** How well an attacker can distinguish between "forget" and "retain" data using membership inference attacks.

**Interpretation:**
- **Lower values (negative is good)** = Better privacy protection
  - AUC close to 0.5 = Random guessing (ideal)
  - Negative values = Better than retrained baseline
- **Higher values (positive)** = Privacy leakage
  - AUC > 0.5 = Attacker can identify training data
  - Model still "remembers" which data was in training

**How it works:**
- Uses perplexity (PPL) to distinguish forget vs. retain vs. holdout sets
- Computes AUC (Area Under Curve) for membership inference
- Compares against retrained baseline (AUC_RETRAIN)
- Formula: `(AUC - AUC_RETRAIN) / AUC_RETRAIN * 100`

**Insight:** This measures **privacy protection** - can attackers detect if data was in training?

---

### 3. **knowmem_f** (KnowMem Forget) - **Lower is Better** ✅
**What it measures:** How well the model can answer questions about knowledge from the "forget" set.

**Interpretation:**
- **Lower values (closer to 0%)** = Better unlearning
  - Model cannot answer questions about forget set knowledge
  - Successful unlearning of semantic knowledge
- **Higher values (closer to 100%)** = Poor unlearning
  - Model can still answer questions correctly
  - Knowledge not fully unlearned

**How it works:**
- Uses few-shot prompting (in-context learning)
- Generates answers to questions about forget set
- Compares with ROUGE-L score
- Range: 0-100% (percentage)

**Insight:** This measures **semantic knowledge** - can the model still use the information even if it can't reproduce it verbatim?

---

### 4. **knowmem_r** (KnowMem Retain) - **Higher is Better** ✅
**What it measures:** How well the model retains knowledge from the "retain" set (utility preservation).

**Interpretation:**
- **Higher values (closer to 100%)** = Better utility preservation
  - Model still performs well on retain set
  - Unlearning didn't damage general capabilities
- **Lower values (closer to 0%)** = Catastrophic forgetting
  - Model lost general knowledge
  - Unlearning too aggressive

**How it works:**
- Same as knowmem_f but for retain set
- Uses few-shot prompting
- ROUGE-L score for answers
- Range: 0-100% (percentage)

**Insight:** This measures **utility preservation** - did unlearning break the model's general capabilities?

---

### 5. **gen** (MMLU - General Knowledge) - **Higher is Better** ✅
**What it measures:** Performance on Massive Multitask Language Understanding benchmark.

**Interpretation:**
- **Higher values** = Better general knowledge retention
- Tests model on diverse academic subjects
- Range: 0-1 (or 0-100% if converted)

**Insight:** Measures **general model quality** after unlearning.

---

### 6. **tru** (TruthfulQA) - **Higher is Better** ✅
**What it measures:** Model's truthfulness and ability to avoid false information.

**Interpretation:**
- **Higher values** = More truthful responses
- Tests if model generates accurate, honest answers
- Range: 0-1 (or 0-100% if converted)

**Insight:** Measures **truthfulness** - does unlearning affect model honesty?

---

### 7. **fac** (TriviaQA) - **Higher is Better** ✅
**What it measures:** Model's ability to answer factual questions.

**Interpretation:**
- **Higher values** = Better factual knowledge
- Tests factual recall and reasoning
- Range: 0-1 (or 0-100% if converted)

**Insight:** Measures **factual knowledge retention**.

---

### 8. **flu** (Fluency) - **Higher is Better** ✅
**What it measures:** How fluent and natural the model's generated text is.

**Interpretation:**
- **Higher values** = More fluent text
- Tests if unlearning damaged language generation quality
- Range: 0-1 (or 0-100% if converted)

**Insight:** Measures **language quality** - is the model still coherent?

---

## 🎯 Key Insights to Look For

### 1. **Trade-off Analysis**
Compare metrics to understand the unlearning trade-off:

```
Good Unlearning:
├── verbmem_f: LOW (0-20%)
├── privleak: LOW/NEGATIVE (-50% to 0%)
├── knowmem_f: LOW (0-20%)
└── knowmem_r: HIGH (80-100%) ← Utility preserved!

Poor Unlearning:
├── verbmem_f: HIGH (50-100%)
├── privleak: HIGH/POSITIVE (0% to +50%)
├── knowmem_f: HIGH (50-100%)
└── knowmem_r: LOW (0-50%) ← Catastrophic forgetting!
```

### 2. **Method Comparison**
When comparing different unlearning methods (e.g., `ga`, `npo`, `ga_gdr`):

**Best Method Characteristics:**
- ✅ Low `verbmem_f` (forgets verbatim text)
- ✅ Low `privleak` (protects privacy)
- ✅ Low `knowmem_f` (forgets semantic knowledge)
- ✅ High `knowmem_r` (preserves utility)
- ✅ High `gen`, `tru`, `fac`, `flu` (maintains quality)

### 3. **Quantization Impact**
If testing with `quantize_4bit=1` or `quantize_8bit=1`:

**Expected Behavior:**
- Quantization may **improve** unlearning metrics (lower verbmem_f, privleak, knowmem_f)
- But may **degrade** utility metrics (lower knowmem_r, gen, tru, fac, flu)
- This is the **"catastrophic failure"** the paper discusses!

**Key Insight:** Quantization can make models "forget" too aggressively, breaking their general capabilities.

### 4. **Corpus-Specific Insights**
Compare `news` vs. `books` corpus:

- **News:** Typically shorter, more factual content
- **Books:** Longer, narrative content with more context

Different corpora may show different unlearning patterns.

---

## 📈 Reading Your Results CSV

Your `output.csv` file contains columns:
```
name, verbmem_f, privleak, knowmem_f, knowmem_r, gen, tru, fac, flu
```

**Example Interpretation:**
```csv
name,verbmem_f,privleak,knowmem_f,knowmem_r,gen,tru,fac,flu
original_target,15.2,-45.3,12.8,85.4,0.65,0.72,0.58,0.81
ga,8.5,-60.2,5.3,78.9,0.62,0.70,0.55,0.79
ga_quantized,2.1,-80.5,1.2,45.2,0.35,0.42,0.28,0.51
```

**Analysis:**
- `ga` vs `original_target`: Better unlearning (lower verbmem_f, privleak, knowmem_f) but slightly lower utility
- `ga_quantized`: Excellent unlearning but **catastrophic utility loss** (knowmem_r dropped from 78.9% to 45.2%)

---

## 🔍 Detailed Analysis Tips

### 1. **Check Intermediate Results**
Look in `temp/{model_name}/` directories:
- `verbmem_f/log.json`: Individual sample scores
- `privleak/auc.json`: Detailed AUC scores for different PPL metrics
- `knowmem_f/agg.json`: Aggregated statistics with confidence intervals

### 2. **Confidence Intervals**
The `agg.json` files contain confidence intervals (CI):
- `rougeL_ci_lo`, `rougeL_ci_hi`: 95% confidence interval
- Use these to assess statistical significance

### 3. **PrivLeak Details**
The `privleak/auc.json` contains multiple metrics:
- `forget_holdout_Min-40%`: Default metric used
- Other metrics (`Min-5%`, `Min-10%`, etc.) test different thresholds
- Lower AUC = Better privacy protection

---

## 🎓 Research Insights

Based on the paper "Catastrophic Failure of LLM Unlearning via Quantization":

### Main Finding:
**Quantization can cause catastrophic failure** where:
1. ✅ Unlearning metrics improve (model "forgets" better)
2. ❌ Utility metrics degrade (model becomes useless)

### Why This Matters:
- Quantization is often used to reduce model size
- But it may not be safe for unlearning applications
- Need to balance unlearning effectiveness vs. utility preservation

### Practical Recommendations:
1. **Test both quantized and full-precision models**
2. **Monitor `knowmem_r` closely** - if it drops significantly, unlearning is too aggressive
3. **Compare multiple methods** - some may preserve utility better
4. **Use corpus-specific evaluation** - different domains may behave differently

---

## 📝 Next Steps

1. **Run full evaluation** (set `MAX_SAMPLES=None`) for accurate results
2. **Compare multiple methods** (ga, npo, ga_gdr, etc.)
3. **Test with/without quantization** to see the catastrophic failure
4. **Analyze detailed logs** in `temp/` directories
5. **Visualize results** using the CSV data

---

## 🚨 Red Flags

Watch out for these warning signs:

1. **`knowmem_r < 50%`**: Model lost too much utility
2. **`privleak > 0%`**: Privacy leakage detected
3. **`verbmem_f > 30%`**: Model still memorizes verbatim text
4. **All metrics = 0.0**: Evaluation may have failed (check logs)

---

## 💡 Questions to Answer

Use your results to answer:

1. **Which unlearning method works best?**
   - Compare verbmem_f, privleak, knowmem_f across methods
   
2. **Does quantization help or hurt?**
   - Compare quantized vs. full-precision models
   
3. **Is there a trade-off?**
   - Plot verbmem_f vs. knowmem_r to see the trade-off curve
   
4. **Which corpus is harder to unlearn?**
   - Compare news vs. books results

---

**Happy Analyzing! 🎉**

