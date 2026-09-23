# Research Implementation Dossier
## Papers 2, 3 and 4 — Source-Grounded Algorithms for Model Steganalysis

**Purpose:** Give a coding/research AI the important implementation-relevant material from the three papers discussed for the model-steganalysis project. This document is a technical research dossier, not a project build prompt.

**Important rule:** Preserve the authors' mechanisms, equations, assumptions, and terminology. Do not silently replace the proposed algorithms with generic heuristics. Where the papers leave an implementation detail unspecified, mark it as unspecified and make the choice configurable/documented rather than pretending it came from the paper.

---

# Paper 2 — Trojan Signatures in DNN Weights

**Authors:** Greg Fields, Mohammad Samragh, Mojan Javaheripi, Farinaz Koushanfar, Tara Javidi  
**Title:** *Trojan Signatures in DNN Weights*

## 1. Security problem

The paper addresses DNN backdoor/trojan attacks in which a model behaves normally on standard inputs but maps inputs containing a trigger to an attacker-selected target class.

The proposed detector is deliberately lightweight:

- no training/test data required by the detector;
- no reverse-engineering of the trigger;
- no expensive iterative optimization;
- analysis is performed on the parameters of the final linear classification layer.

The central hypothesis is:

> Trojan insertion often produces a detectable statistical signature in the final classification layer, particularly an unusually large average weight associated with the target class.

## 2. Mechanistic basis

Let the final classification layer have weight matrix:

    W

where each row corresponds to a class.

For a classification model with `c` classes, row `i` is:

    W_i = [W_i,1, W_i,2, ..., W_i,d]

where `d` is the dimensionality of the penultimate representation.

The paper's mechanism is based on the observation that a trojan target class tends to accumulate positive contributions from feature representations associated with many source classes.

Therefore the target-class row can become an outlier.

This is important for our project because this signal is:

- parameter-based;
- interpretable;
- inexpensive;
- attack-mechanism-specific rather than a generic collection of statistics.

## 3. Per-class statistic

For every class row, compute its average weight:

    w_i = (1/d) * sum_j W_i,j

Then sort the class averages:

    w_i1 <= w_i2 <= ... <= w_ic

The candidate target row is the row having the largest average weight:

    w_ic

## 4. Dixon Q statistic

The paper applies Dixon's Q-test for a single outlier in a small sample.

The Q statistic used is:

    Q = |w_ic - w_i(c-1)| / (w_ic - w_i1)

Interpretation:

- numerator = gap between the largest and second-largest row averages;
- denominator = total range between the largest and smallest row averages;
- a large Q indicates that the largest row is unusually separated from the rest.

The paper then compares Q against a tabulated Dixon critical value for the relevant sample size and significance level.

Example explicitly discussed by the paper:

- for an 8-class model, `Q > 0.468` gives an outlier conclusion under the cited tabulated test setting.

Do NOT treat `0.468` as a universal detector threshold. Critical values depend on the Dixon-test configuration.

## 5. Detection procedure

Implementation flow:

    model
      ↓
    locate final linear classification layer
      ↓
    read weight matrix W
      ↓
    compute mean of each class row
      ↓
    sort class means
      ↓
    identify largest row
      ↓
    compute Dixon Q
      ↓
    compare with appropriate tabulated critical value
      ↓
    return:
      - Q statistic
      - candidate target class
      - row means
      - critical value
      - significance decision
      - evidence/explanation

The paper's detector is naturally a module in the project's weight-space evidence layer.

## 6. Why it belongs in a multi-evidence scanner

This detector should NOT automatically become the overall final verdict.

Instead:

    TrojanSignatureEvidence
        Q statistic
        target row index
        row mean distribution
        statistical significance
        confidence/evidence strength

These results become one evidence signal in the project's later fusion stage.

A model can be suspicious for other reasons even when the final-layer signature is weak.

## 7. Empirical behavior and limitation

The paper reports that increasing the poisoned-data proportion generally strengthens the weight signature. It also reports that the target-row average can remain an outlier across several trigger types and datasets.

An important failure case is the reported age-detection example:

- target-row weight was increased;
- however, the change was relatively small;
- the target row was already unusually small;
- the detector failed to identify it reliably.

This is important for our evaluation: the module must not be represented as a universal trojan detector.

## 8. Adaptive attacker experiment

The authors implement an adaptive attack intended to suppress the statistic used by the detector.

They modify the normal cross-entropy loss:

    L_reg = L_CE + γ [ E[W_t] - E[W] ]

where:

- `W_t` = weights in the target-class row;
- `E[W_t]` = average target-row weight;
- `E[W]` = average of all weights in the final weight matrix;
- `γ` = regularization strength.

This is an attacker-side/evaluation formulation, not the normal detector.

Purpose:

    reduce target-row mean
            ↓
    reduce Q statistic
            ↓
    make the target row less obvious

Reported trade-off:

Increasing regularization masked the weight signature, but also reduced clean-model accuracy and trigger effectiveness. The paper therefore demonstrates that detector-aware evasion should be part of robustness evaluation.

## 9. Implementation requirements

Implement at least:

### Detector module

Inputs:
- supported model;
- final classification layer weight matrix.

Outputs:
- class-row means;
- sorted means;
- candidate outlier row;
- Dixon Q;
- relevant critical value;
- pass/fail/significance result;
- machine-readable evidence object.

### Evaluation/adaptive attack module

Optional later experiment:
- train a trojaned model with the regularized loss;
- vary `γ`;
- measure:
  - Q statistic;
  - clean accuracy;
  - trigger efficacy / ASR.

### Testing

Tests should verify:
- row means are calculated correctly;
- sorting does not lose original class index;
- Q formula is numerically correct;
- degenerate range is handled;
- critical values are selected by sample size/configuration;
- adaptive regularization affects the target-row statistic in the expected direction.

---

# Paper 3 — Steganalysis of Neural Networks Using Implicit Features

**Authors:** Jiaming Cao, Zichi Wang, Yunlong Hao, Xinpeng Zhang  
**Title:** *Steganalysis of neural networks using implicit features*  
**Journal:** Cybersecurity, 2026.

## 1. Security problem

This paper addresses black-box steganalysis of neural networks.

Unlike white-box methods that inspect model parameters, this method attempts to detect hidden information without access to internal weights.

The central idea:

> Feed a fixed sequence of images into a target classification network and use the model's output probabilities as an implicit behavioral representation of the model.

The resulting representation is then given to a learned steganalysis network.

## 2. High-level pipeline

Training:

    plain models + secret/stego models
                ↓
    fixed image sequence
                ↓
    model output probabilities
                ↓
    implicit feature matrix D_k
                ↓
    preprocessing
                ↓
    steganalysis CNN
                ↓
    plain/secret classifier

Detection:

    unknown model
         ↓
    same fixed image sequence
         ↓
    output probabilities
         ↓
    same preprocessing
         ↓
    trained steganalysis network
         ↓
    probability / decision

## 3. Notation

The paper defines:

- `N` = number of images in the fixed sequence.
- `{I1, I2, ..., IN}` = fixed image sequence.
- `M` = number of classification models.
- `{NET1, ..., NETM}` = classification networks.
- `Dk` = implicit feature representation of model `NETk`.
- `Lk` = label for `Dk`.
- `Rk` = steganalysis output for `Dk`.
- `C` = maximum embedding capacity.
- `β` = weight for embedding loss in the model-generation process.

For a `g`-class classification network:

    F_ki ∈ R^g

is the softmax output vector for image `I_i`.

Therefore:

    D_k = {F_k1, F_k2, ..., F_kN}

has shape:

    N × g

## 4. Construction of the fixed sequence

The paper constructs the fixed image sequence by selecting:

    N/g images from each of the g categories

for the image-classification setting used in the experiments.

The selected fixed sequence is not used for training the classification networks.

This separation is important because otherwise the feature-extraction probes could leak training information.

## 5. Implicit feature extraction

For each model:

    for i = 1 ... N:
        F_ki = softmax(NET_k(I_i))

Then:

    D_k = [F_k1, F_k2, ..., F_kN]

The paper treats `D_k` as the model's implicit features.

Labels:

    L_k = 0  if NET_k is plain
    L_k = 1  if NET_k contains secret data

## 6. Feature preprocessing

In the paper's main experiments:

    N = 1000

and the classification tasks have:

    g = 10 classes

so:

    D_k shape = 1000 × 10

The paper compares two preprocessing forms.

### Method A — reshape into 100 × 100

Every 10-dimensional feature vector is arranged as one row:

    1000 × 10 → 100 × 100

### Method B — preserve the 1000 × 10 structure

Keep the output directly as:

    1000 × 10

The paper constructs `model-06` and `model-06-long` for comparison and reports better performance for `model-06-long`.

Therefore the paper's selected representation is:

    1000 × 10

This is important: do not silently replace this with an arbitrary feature pooling scheme when reproducing the paper.

## 7. Steganalysis network

The paper formulates detection as binary classification.

The detector is:

    R_d = f_theta(D_d)

where:

- `D_d` = implicit features from the model being detected;
- `f_theta` = trained steganalysis network;
- `R_d` is expected to approach:
  - `0` for plain models;
  - `1` for secret models.

The paper uses a CNN-based steganalysis model.

The described building blocks include:

### Convolution

    (F^l)_ij =
        sum_u sum_v (F^(l-1))_u,v (W^l)_i,j,u,v
        + b_i,j

### ReLU

    ReLU(x) = max(0, x)

### Average pooling

    (F^l)_ij =
        1/(w*h)
        * sum_p sum_q
          (F^(l-1))_(i*s+p),(j*s+q)

### Fully connected layer

    F^l = W * F + b

### Binary cross entropy

The paper writes its binary loss as:

    loss = -RlogL - (1-R)logL

The typeset equation appears to have a notation inconsistency; implementation should use the intended binary cross-entropy between prediction `R` and binary label `L`, rather than copying an obvious symbolic typo literally.

## 8. Selected detector architecture

The paper evaluates several steganalysis networks.

Its selected model is:

    model-06-long

The paper describes it as using a convolutional layer, two residual blocks, global average pooling, and a fully connected output layer with sigmoid activation.

The reason for selecting this architecture is an empirical trade-off: deeper alternatives can capture additional relationships but were observed to be less robust.

## 9. Training configuration reported by the paper

The reported detector training setup includes:

- training epochs: `30`
- learning rate: `0.001`
- batch size: `32`
- loss: binary cross entropy
- optimizer: Adam
- train/test split: `80% / 20%`

These are paper-reported settings and should be configuration values for a reproduction rather than hard-wired assumptions for every future dataset.

## 10. Model dataset used by the paper

The authors create many plain/secret model pairs.

Their experimental categories cover:

- different classification architectures;
- different classification datasets/tasks;
- different embedding capacities.

The paper reports:

- 3 architecture families in its experiments: AlexNet, ResNet, CapsNet;
- datasets/tasks including MNIST, CIFAR-10 and an ImageNet subset;
- 5 embedding-capacity settings;
- 45 categories in one construction;
- 1000 plain models and 1000 secret models per category in the large-scale experiment.

These exact dataset/model counts are useful for understanding the paper's training scale, but the project should not require reproducing all 45,000 category-model combinations before the method can be tested.

## 11. Ablation findings

### Number of training models M

The paper evaluates different values including:

    M = 100, 500, 1000

Increasing the number of training models provides more diverse feature information.

### Number of fixed images N

The paper evaluates:

    N = 100, 500, 1000

Increasing `N` improves steganalysis accuracy because more probes capture finer behavioral characteristics.

## 12. Important limitations stated by the paper

The authors identify important constraints:

1. A large number of classification models are needed in advance to construct the training dataset.
2. The paper used 1000 models per category, creating substantial computational/time cost.
3. There is currently no quantitative standard for choosing the fixed image sequence; selection is primarily based on image diversity.
4. The current scheme is designed for classification networks and is not demonstrated for arbitrary neural-network task types.

These limitations matter directly to our project architecture.

## 13. Integration into our project

This module should be an optional behavioral evidence branch:

    model executable?
          |
        yes
          ↓
    fixed probe set
          ↓
    output probability matrix
          ↓
    implicit feature preprocessing
          ↓
    trained steganalysis detector
          ↓
    behavioral evidence score

When safe/executable inference is unavailable, the scanner should continue using weight-space evidence instead of failing the entire scan.

The paper-derived behavioral score should be kept separate from the other evidence sources until the project's fusion stage.

---

# Paper 4 — Disarming Steganography Attacks Inside Neural Network Models

**Author:** Ran Dubin  
**Title:** *Disarming Steganography Attacks Inside Neural Network Models*

## 1. Security problem

The paper studies hiding data or malware in neural-network weights using low-order bits of floating-point parameters.

The model can continue to perform its ML task while its weights act as a storage medium.

The paper focuses on TensorFlow/PyTorch-style FP32 parameters and describes both:

- LSB steganographic attack mechanisms;
- zero-trust countermeasures / CDR.

For our detector project, the most important part is the attack methodology because it gives us controlled manipulated models for evaluation.

## 2. Floating-point representation used

The paper treats the model weights as IEEE-754 32-bit floating-point values.

The 32 bits consist of:

- 1 sign bit;
- 8 exponent bits;
- 23 mantissa bits.

Thus, for FP32:

    1 + 8 + 23 = 32 bits

The paper emphasizes that changing the lower mantissa bits can produce very small changes to the numerical floating-point value.

Illustrative values discussed in the paper:

    0x3C000000

versus:

    0x3CFFFFFF

and:

    0x3C0000FF

The paper uses these to show that changing lower-order bits can have much smaller numerical impact than changing more significant mantissa bits.

## 3. Model storage capacity

For a layer with:

- `m` neurons;
- `n` incoming weights per neuron;
- one bias per neuron;

the paper describes:

    m(n + 1)

parameters.

For FP32:

    size = 4 * m * (n + 1) bytes

This explains why large models provide a large potential storage channel.

## 4. Baseline LSB attack family

The paper evaluates three LSB substitution attacks.

These should be implemented in our evaluation/attack-generation layer as controlled attack generators.

### FMLA — Full Mantissa LSB Attack

Replace the entire 23-bit mantissa of the targeted FP32 float with payload bits.

The paper's ResNet-101 example reports an overall capacity of up to approximately 116 MB when using 23-bit substitution across the considered Conv2D neurons.

This is the strongest and most destructive of the three baseline variants.

### HMLA — Half Mantissa LSB Attack

Replace the lowest 12 mantissa bits.

Reported ResNet-101 example:

    approximately 60 MB capacity

The paper observes that this type is considerably less visible numerically than changing the whole mantissa.

### HBLA — Half Byte LSB Attack

Replace the lowest 4 mantissa bits.

Reported ResNet-101 example:

    approximately 20 MB capacity

The paper observes that this lower-bit attack can often have little observable effect on model accuracy, making it a useful stealth-oriented evaluation case.

## 5. Security interpretation for our project

The three attacks provide a useful severity ladder:

    FMLA
      ↓
    stronger numerical/model-performance disturbance

    HMLA
      ↓
    more subtle

    HBLA
      ↓
    highly low-bit-focused / subtle

This is useful for controlled experiments such as:

    clean
    ↓
    HBLA
    ↓
    HMLA
    ↓
    FMLA

The project should measure detection rate against each class separately rather than reporting only one combined attack score.

## 6. Accuracy/performance observations

The paper reports that:

- FMLA can cause significant model-performance degradation because it changes more significant mantissa information.
- HMLA and HBLA are often much less noticeable.
- Some models such as VGG16 showed greater resilience to FMLA than others.
- Smaller payloads modify fewer weights and can therefore produce smaller accuracy degradation.
- The paper specifically notes cases where HBLA produced no observed accuracy degradation.

These observations are important for false-negative stress testing.

## 7. CDR / disarming methods

The paper additionally proposes zero-trust countermeasures.

These are not the primary Phase-1 detection algorithm, but they should be retained in the research dossier because they are relevant to future mitigation.

### Full LSB Prevention (FLP)

Replace all 23 mantissa LSB bits for the relevant Conv2D neurons with random bits.

This is highly aggressive.

The paper reports that it can produce significant model-performance damage in many cases.

### K-LSB Random Bits Prevention (K-LRBP)

For each targeted neuron, randomly select `k` bits to replace.

The paper evaluates:

    k = 1, 5, 10

This attempts to reduce the hidden payload while limiting model degradation.

### Quantization / Qint8

The paper also evaluates 8-bit quantization.

The paper gives:

    x_q = Clip(Round(x_f / scale))

with:

    scale = (2 * a_max) / 256

and:

    a_max = max(abs(x_f))

where `Clip` restricts the quantized values to the intended signed 8-bit range.

The paper discusses the distinction between ordinary floating-point representation and reduced-bit integer representation.

## 8. Scope decision for our project

For the model-steganalysis scanner:

### Phase-1 / core evaluation

Implement:

- FMLA attack generator;
- HMLA attack generator;
- HBLA attack generator.

Use them to produce controlled evaluation datasets.

### Not primary Phase-1 detection

Do not make FLP/K-LRBP/Qint8 the main detector.

### Future mitigation extension

The CDR methods can later be placed in:

    suspicious model
          ↓
    mitigation/sanitization
          ↓
    reconstructed / disarmed model
          ↓
    validation

This aligns with the project architecture's future sanitization extension.

## 9. Evaluation requirements from this paper

For every generated attacked model, record at least:

- source clean-model hash;
- architecture;
- affected layers;
- attack type;
- number of modified mantissa bits;
- payload size;
- percentage of weights modified;
- numerical weight differences;
- clean accuracy;
- attacked accuracy;
- attack validity / extraction validity where applicable.

The detector should not be evaluated only on models that retain identical accuracy. Subtle attacks and visibly damaging attacks should both be included.

---

# Cross-Paper Implementation Map

These three papers are complementary rather than interchangeable.

## Evidence module A — Trojan Signature

Source:

    Trojan Signatures in DNN Weights

Mechanism:

    final-layer row means
          ↓
    Dixon Q-test
          ↓
    target-class outlier evidence

Nature:

    deterministic statistical / mechanistic

---

## Evidence module B — Implicit Behavioral Features

Source:

    Steganalysis of Neural Networks Using Implicit Features

Mechanism:

    fixed probe sequence
          ↓
    output probability vectors
          ↓
    1000 × g implicit feature representation
          ↓
    model-06-long steganalysis network
          ↓
    behavioral evidence

Nature:

    learned black-box behavioral evidence

---

## Attack/evaluation module C — LSB weight attacks

Source:

    Disarming Steganography Attacks Inside Neural Network Models

Mechanism:

    FP32 mantissa-bit substitution
          ↓
    FMLA / HMLA / HBLA
          ↓
    manipulated model
          ↓
    scanner evaluation

Nature:

    controlled attack-generation benchmark

---

# How the Three Papers Fit the Larger Scanner

The papers should not be merged into one giant detector.

A cleaner architecture is:

    MODEL
      |
      +----------------------------+
      |                            |
      v                            v
  Weight-space                 Behavioral
  analysis                     analysis
      |                            |
      |                            +--> Implicit Features
      |
      +--> Trojan Signature
      |
      +--> other evidence modules
      |
      v
  Evidence normalization
      |
      v
  Multi-evidence fusion
      |
      v
  Model-level risk assessment
      |
      v
  Explainable report

Separately:

    CLEAN MODEL
         |
         +--> FMLA
         +--> HMLA
         +--> HBLA
         |
         v
    CONTROLLED ATTACK DATASET
         |
         v
    EVALUATION OF ALL DETECTOR BRANCHES

This separation is important:

- Paper 2 gives an actual detection signal.
- Paper 3 gives an actual detection signal.
- Paper 4 gives controlled attack cases and future mitigation mechanisms.

---

# Reproduction Rules

When implementing these papers:

1. Preserve the source terminology.
2. Preserve equations where the paper provides equations.
3. Do not replace a paper-specific method with a generic statistic and call it the same algorithm.
4. Keep paper-derived modules isolated and attributable.
5. Make paper parameters configurable rather than hard-coded when practical.
6. Record which implementation details are:
   - explicitly specified by the paper;
   - inferred from equations/figures;
   - engineering adaptations required for the new project.
7. Add unit tests for mathematical/statistical components.
8. Add reproduction tests using synthetic inputs where an exact expected result can be calculated.
9. Do not claim that a reproduced method achieves the paper's published accuracy until it has been independently evaluated.
10. Keep the authors' methods distinct from the project's own multi-evidence fusion contribution.

# Known Source Limitations That Must Not Be Hidden

## Trojan Signatures

The final-layer signature can become too weak for the Dixon Q-test to identify the target class reliably. The reported age-detection case is an explicit example.

Adaptive regularization can suppress the specific statistic but introduces performance degradation and a different weight-distribution signature.

## Implicit Features

The method is currently demonstrated for classification networks.

It requires a sufficiently large training collection of plain/secret models.

The paper reports no quantitative standard for selecting the fixed image sequence; image diversity is used.

Training many models is computationally expensive.

## Disarming Attacks

The most subtle LSB attacks may barely change normal model accuracy, meaning accuracy-preservation alone cannot establish that a model is safe.

More aggressive bit substitution can heavily damage model function.

The paper's CDR techniques have their own accuracy/security trade-offs and should not be confused with detection.

---

# Source Attribution

Use the original papers as the authoritative source for these implementations:

1. Fields et al., *Trojan Signatures in DNN Weights*.
2. Cao et al., *Steganalysis of neural networks using implicit features*, Cybersecurity (2026).
3. Dubin, *Disarming Steganography Attacks Inside Neural Network Models*.

This document is an implementation-oriented extraction of the relevant material from those papers; it is not a replacement for the papers themselves.
