---
name: reaxff-literature
description: Published ReaxFF parameterization practice distilled for decisions - force-field branches and compatibility, training-set content, weights/sigmas, which parameters to fit and in what order, parameter bounds, optimizers (SOPPE, MCFF, CMA-ES, GA, gradient/JAX), restarts and noise, charge models (EEM/QEq/ACKS2) and their pathologies, bond-order discontinuities, short-range repulsion, validation and attainable accuracy - with verified citations. Use before choosing or justifying any fitting setting, bound, weight, optimizer or validation criterion, and when writing methods sections.
---

# ReaxFF parameterization literature (for agents)

Distilled from the papers below; keys like **[Sh19]** refer to the table.
Every citation was checked against Crossref (Oct 2026). Statements are
paraphrased from the full text unless marked *(abstract only)* or
*(unverified)*. When you rely on a point, cite it (author, year, DOI) in
what you write. If you deviate from published practice, say so and why.
Users can keep PDFs in `reaxff_papers/` (gitignored); this file is the
tracked summary.

| Key | Paper | DOI | Use it for |
|---|---|---|---|
| vD94 | van Duin, Baas, van de Graaf, *J. Chem. Soc. Faraday Trans.* 90, 2881 (1994) | 10.1039/ft9949002881 | successive one-parameter parabolic search (SOPPE) |
| vD01 | van Duin, Dasgupta, Lorant, Goddard, *JPCA* 105, 9396 (2001) | 10.1021/jp004368u | the ReaxFF functional form; hydrocarbon training set |
| vD03 | van Duin et al., *JPCA* 107, 3803 (2003) | 10.1021/jp0276303 | Si/SiO: early multi-element training-set design |
| C08 | Chenoweth, van Duin, Goddard, *JPCA* 112, 1040 (2008) | 10.1021/jp709896w | C/H/O combustion branch; transition states, MD validation |
| S16 | Senftle et al., *npj Comput. Mater.* 2, 15011 (2016) | 10.1038/npjcompumats.2015.11 | review: branches, fitting practice, charge models |
| L13 | Larsson, van Duin, Hartke, *J. Comput. Chem.* 34, 2178 (2013) | 10.1002/jcc.23382 | GA fitting of SiOH; ranges matter most |
| I13 | Iype, Hütter, Jansen, Nedea, Rindt, *J. Comput. Chem.* 34, 1143 (2013) | 10.1002/jcc.23246 | Monte Carlo / annealing optimizer (MCFF) *(abstract + SCM docs)* |
| D15 | Dittner, Müller, Aktulga, Hartke, *J. Comput. Chem.* 36, 1550 (2015) | 10.1002/jcc.23966 | parallel global optimization (ogolem + PuReMD) *(abstract)* |
| MH16 | Müller, Hartke, *JCTC* 12, 3913 (2016) | 10.1021/acs.jctc.6b00461 | disulfide FF from multireference data; bounds practice (SI) |
| Sh19 | Shchygol, Yakovlev, Trnka, van Duin, Verstraelen, *JCTC* 15, 6799 (2019) | 10.1021/acs.jctc.9b00769 | **optimizer benchmark (CMA-ES/MCFF/GA) and guidelines** |
| FW19 | Furman, Wales, *JPCL* 10, 7215 (2019) | 10.1021/acs.jpclett.9b02810 | bond-order cutoff discontinuities, tapering |
| FW20 | Furman, Wales, *JCP* 153, 021102 (2020) | 10.1063/5.0013906 | further discontinuities (lone pairs, SBO2, torsions) |
| NB19 | Nakata, Bai, *J. Comput. Chem.* 40, 2000 (2019) | 10.1002/jcc.25841 | ML-assisted search; equal fits, different parameters |
| P21 | Komissarov, Rüger, Hellström, Verstraelen, *JCIM* 61, 3737 (2021) | 10.1021/acs.jcim.1c00333 | **ParAMS: loss, sigmas, bounds, sensitivity, early stopping** |
| Se21 | Sengul et al., *npj Comput. Mater.* 7, 68 (2021) | 10.1038/s41524-021-00534-4 | INDEEDopt: DNN surrogate of the loss |
| K22 | Kaymak, Rahnamoun, O'Hearn, van Duin, Merz, Aktulga, *JCTC* 18, 5181 (2022) | 10.1021/acs.jctc.2c00363 | JAX-ReaxFF: gradient-based fitting |
| GW22 | Freitas Gustavo, Verstraelen, *J. Cheminform.* 14, 7 (2022) | 10.1186/s13321-022-00581-z | GloMPO: managing many optimizer runs |
| FH23 | Freitas Gustavo, Hellström, Verstraelen, *JCTC* 19, 2557 (2023) | 10.1021/acs.jctc.2c01320 | sensitivity analysis (HSIC) before fitting |
| Du24 | Dumortier, Chizallet, Creton, de Bruin, Verstraelen, *JCTC* 20, 3779 (2024) | 10.1021/acs.jctc.3c01009 | modern protocol: 40 CMA-ES seeds, bounds, DFT-MD validation |
| DA24 | De Angelis, Cappabianca, Fasano, Asinari, Chiavazzo, *Sci. Rep.* 14, 978 (2024) | 10.1038/s41598-023-50978-5 | training on MD/strained/defect frames with forces |
| Yo21 | Yoo, Sakano, Desai, Islam, Liao, Strachan, *npj Comput. Mater.* 7, 9 (2021) | 10.1038/s41524-020-00484-3 | MD -> DFT -> refit loop (for a neural FF, seeded from ReaxFF MD) |
| M18 | Mishra et al., *npj Comput. Mater.* 4, 42 (2018) | 10.1038/s41524-018-0098-3 | multi-objective GA; Pareto front as uncertainty ensemble |
| M86 | Mortier, Ghosh, Shankar, *JACS* 108, 4315 (1986) | 10.1021/ja00275a013 | EEM |
| RG91 | Rappé, Goddard, *J. Phys. Chem.* 95, 3358 (1991) | 10.1021/j100161a070 | QEq |
| V11 | Verstraelen et al., *JCTC* 7, 1750 (2011) | 10.1021/ct200006e | charge-only costs cannot fix EEM/SQE parameters |
| V13 | Verstraelen, Ayers, Van Speybroeck, Waroquier, *JCP* 138, 074108 (2013) | 10.1063/1.4791569 | ACKS2 |
| Ko22 | Koski et al., *JCTC* 18, 580 (2022) | 10.1021/acs.jctc.1c00975 | ACKS2 vs QEq water in fields |
| lg11 | Liu, Liu, Zybin, Sun, Goddard, *JPCA* 115, 11016 (2011) | 10.1021/jp201599t | ReaxFF-lg dispersion |
| Ca20 | Caro, Csányi, Laurila, Deringer, *PRB* 102, 174201 (2020) | 10.1103/PhysRevB.102.174201 | too-soft short-range repulsion in ReaxFF-C |
| Ba13 | Bauschlicher et al., *JPCA* 117, 11126 (2013) | 10.1021/jp408113w | ReaxFF vs CCSD(T) accuracy benchmark |
| Ak12 | Aktulga, Fogarty, Pandit, Grama, *Parallel Comput.* 38, 245 (2012) | 10.1016/j.parco.2011.08.005 | the LAMMPS/PuReMD ReaxFF implementation |

## 1. Starting point: branches and compatibility

- ReaxFF parameter sets come in **branches** that are internally
  transferable: the *combustion* branch (built on C08) and the *aqueous*
  branch, created because the combustion set describes gas-phase water but
  fails for liquid water; it uses different O/H atom and bond parameters
  **[S16]** (later water branch: Zhang & van Duin 2017, 10.1021/acs.jpcb.7b02548).
- Within one branch, combining elements needs only the **bond and angle
  terms between the newly combined elements** to be fitted; moving between
  branches needs much more refitting (combustion -> aqueous is the easier
  direction) **[S16]**. Parameter sets cannot simply be combined at will.
- There is no atom typing: one oxygen type for all chemistries **[S16]**.
- Start from the closest published force field of the right branch; keep
  established blocks fixed for compatibility. Shchygol et al. note that
  ~50 free parameters is typical and the rest stay at historical values for
  backward compatibility **[Sh19]**. Freeing many more (191 in [L13]) breaks
  compatibility with other sets.
- Toolkit: `ffield-merge` (base + donors + `--analog`), whose report lists
  differing general parameters (a sign of another branch).

## 2. Training-set content

- Classic content, per element combination: bond-dissociation curves (other
  coordinates relaxed under a restraint), valence-angle and torsion scans,
  equilibrium geometries, charges, reaction energies including transition
  states, equations of state of condensed phases, heats of formation
  **[vD01, C08, S16]**; van Duin's training notes add under/over-coordinated
  structures. Chenoweth 2008 added transition states and reactive systems for
  high-temperature oxidation and validated with NVT-MD of hydrocarbon/O2
  mixtures **[C08]**.
- Modern fits add **forces** on distorted and MD frames: De Angelis et al.
  used Materials Project cells, strains from -12.5% to +23.5%, defects and
  AIMD frames at 300/500 K (>3000 entries with energies, forces, charges)
  **[DA24]**; Dumortier et al. validated on DFT-MD snapshots **[Du24]**; the
  disulfide set of [MH16] has 4401 force items of 4875.
- **Only unambiguous geometry optimizations** belong in a training set:
  rigid molecules, reaction products as separate entries rather than one
  complex, and enough optimization steps. In [Sh19] the loss noise came from
  8 of 309 (silica) and 11 of 4875 (disulfide) items: flexible molecules with
  several conformers, a complex with two binding sites ~20 kcal/mol apart, a
  quartz cell limited to 5 geometry steps, unconverged dihedrals. A <0.5%
  parameter change could move the loss by ~1000.
- Converge geometry optimizations tightly: [Sh19] used max force
  0.1 kcal/mol/Å and <= 3000 steps ("relaxing convergence ... deteriorates the
  numerical noise"); [Du24] allowed 500 steps and found 50-200 step limits
  overfit to the step limit.
- Include short contacts and compressed states: too-soft short-range
  repulsion makes energetic atoms penetrate too deep **[Ca20]**; ReaxFF fits
  without such data can collapse at high temperature or pressure.
- One level of theory and one reference-energy convention for all data;
  inconsistent references limit what any optimizer can reach **[Sh19]**.

## 3. Loss, sigmas, weights

- Loss = Σ w·((pred − ref)/σ)² (classic ReaxFF: the trainset "weight" is
  really an accuracy σ; ParAMS: SSE, the default and recommended loss)
  **[vD94 manual, Sh19, P21]**. Dihedral residuals are periodic **[Sh19]**.
- Array items: ParAMS splits a scalar weight over the components **[P21]**
  (this toolkit does the same: one force set counts like one item).
- Typical accuracies σ: classic sets use ~2 kcal/mol for heats of formation,
  1.5-2 kcal/mol for reaction energies, 0.01-0.02 Å for bonds, ~1° for
  angles, 0.1 e for charges (ReaxFF manual 2002). ParAMS extractor defaults:
  energy 0.002 Ha (1.25 kcal/mol), forces 0.003 Ha/bohr, charges 0.1 e,
  distance 0.05 Å, angle/dihedral 2°, cell lengths 0.05 Å, cell angles 2°
  **[P21]**.
- Charges are not observables, and charge-only costs cannot determine EEM
  parameters **[V11]**; weight them loosely and fit with other data.
  Population schemes differ; Trnka et al. fitted to electrostatic potentials
  instead (10.1021/acs.jctc.7b00870).

## 4. What to fit, in which order, how many

- Sequential practice: charges (EEM χ, η, γ) -> bond orders and bond
  energies -> valence angles -> over/under-coordination -> reactions -> EOS,
  then joint refinement (van Duin's Si/SiO training narrative; [vD01, C08,
  S16]). The order is practice, not a published prescription.
- Freeze general parameters, masses, valencies, switches, unused slots, the
  bond-order cutoff and taper radii. ParAMS' ZnS example froze general,
  atom, π-bond parameters and anything without training data, and kept
  zero-valued parameters at zero **[P21]**. The ParAMS water tutorial fitted
  only the 12 H-O bond parameters; π parameters were not useful there.
- **Sensitivity first.** ParAMS' disulfide refit scanned all 701 parameters
  one at a time and kept the 35 most sensitive, of which only 16 overlapped
  the 87 the original authors fitted **[P21]**; fitting a small sensitive
  subset converges faster to comparable loss with less overfitting
  **[FH23]**. Rule of thumb from the ParAMS docs: with n training items, fit
  no more than ~sqrt(n) parameters at a time.
- JAX-ReaxFF had to freeze 18 bond-order parameters whose gradients reached
  ~1e17 **[K22]**: bond-order parameters make the loss very rough.

## 5. Bounds

- **Ranges matter more than the optimizer**: GA tuning hardly mattered,
  8x-enlarged ranges performed terribly, and re-deriving ranges from the best
  solutions recovered expert quality **[L13]**.
- [Sh19] used the intervals published with each training set; "educated"
  guesses took the median of historical values of the same parameter
  (same elements first). [Du24]: historical range widened to include ±20%
  around the start ("no established defaults for the parameter bounds").
  ParAMS default: x0 ± 20%·|x0|, zero -> ±1 **[P21]**.
- Physical limits: non-negative energies/radii, bond-order exponents and
  p_val4 >= 1, 0 <= θ0 < 180 **[P21]**. If a fit pushes a parameter with
  physical meaning (a radius) to unrealistic values, fix it or narrow its
  interval **[Sh19]**; [MH16] narrowed the S vdW radius from 1.5-2.5 Å to
  1.7-2.0 Å after values above 2.2 Å made MD unstable.
- ParAMS imposes r0σ >= r0π >= r0ππ by default **[P21]**, but published sets
  violate it (Chenoweth 2008 carbon has r_pi < r_pipi), so this toolkit only
  keeps orderings the starting force field satisfies.

## 6. Optimizers

- **SOPPE** [vD94]: per parameter evaluate p−δ, p+δ (δ relative to p), fit a
  parabola through the three losses, move to its minimum within an
  extrapolation limit and the bounds (keep the best of three if the parabola
  opens downward); one pass down the params file. Transparent, but blind to
  parameter correlation and prone to local minima **[S16]**; most published
  force fields were made with it "in conjunction with multiple restarts and a
  large amount of human experience" **[L13]**. Use for polishing.
- **MCFF** (Metropolis + annealing) [I13]; [Sh19] settings: move 10% of
  parameters by up to ±1% of their range, step ×/÷1.05 to hold acceptance at
  50-70%, β from sqrt(N/(2·C1·L0)) to sqrt(N/(2·C2)), C1 = 1, C2 = 5; 3×3k
  restarts better than one 9k run. "Overall unimpressive" **[Sh19]**.
- **CMA-ES** (recommended default): population 4 + floor(3 ln N), initial σ
  = range/4 in normalized parameters, out-of-bounds points get the worst loss
  **[Sh19]**. Lowest loss in 2 of 3 benchmark sets, but not reproducible
  run-to-run; GA (pool 500) was the most reproducible but slow **[Sh19]**.
  ParAMS' default optimizer; its ReaxFF case used population 36, σ 0.3,
  ±20% bounds, 9 repeats **[P21]**; [Du24] used 40 seeds × 1000 iterations,
  σ0 = 0.2.
- **Never trust one run.** "Perform 10 independent CMA-ES runs and take the
  best"; better, run many short runs and continue only the best (after 45k
  evaluations, 90% of runs improved < 4%; a few thousand evaluations per
  short run would have sufficed); the starting guess mattered little - the
  best of 150 runs always came from a random start **[Sh19]**. GloMPO
  manages such portfolios automatically **[GW22]**.
- **Many near-degenerate minima** with very different parameters [L13, Sh19,
  NB19]: choose among them with a held-out test set (early stopping
  analogue), and treat their spread as an uncertainty estimate - add
  training data where they disagree **[Sh19]**. Equal-loss sets can differ in
  MD stability: some of Nakata & Bai's well-fitting sets failed to keep a
  crystal stable at 2000 K **[NB19]**.
- **Gradient-based** (JAX-ReaxFF): L-BFGS-B on single points at fixed
  relaxed geometries, re-relaxing in an outer loop, 10 multistarts, noise
  added when stuck; minutes on a GPU vs days for the GA on the [Sh19] sets
  **[K22]**. EEM only, no lg, no cell optimization in the public code.
- Surrogates (DNN [Se21], random forest [NB19]) and multi-objective GA
  [M18] exist; INDEEDopt found minima that satisfied heats of formation and
  EOS simultaneously, which were anti-correlated in the conventional fit
  **[Se21]**.

## 7. Charges

- ReaxFF charges come from EEM/QEq **[M86, RG91]**; Coulomb is shielded,
  1/(r³ + γ⁻³)^(1/3), and tapered at 10 Å (C08 SI).
- Keep η > 7.2·γ for every element, or the hardness matrix loses positive
  definiteness at short range and charges run away (polarization
  catastrophe) - SCM ReaxFF manual. LAMMPS uses 2× the ffield η internally.
- EEM's worst flaw: unrestrained long-range charge transfer (small charges
  on isolated molecules, transfer between dielectric phases) **[S16]**.
  ACKS2 fixes it **[V13, Ko22]** but needs reparameterization (EEM parameters
  are a good start) **[S16]**; it needs `bcut_acks2` (atom item 23) and the
  bond softness (general parameter 35).
- LAMMPS QEq/ACKS2 conserve zero total charge: charged training structures
  cannot be reproduced there.

## 8. Discontinuities and numerical behaviour

- Bonds, angles, torsions and H-bonds switch off abruptly at hard bond-order
  cutoffs (0.001 for bonds, 0.0001 angles/torsions, 0.01 H-bonds in the
  reference code). Consequences: cusps in dissociation curves (N2 near
  2.5 Å), geometry optimizations that fail to converge (5 of 7 systems),
  wrong minima, NVE energy drift **[FW19]**. Acceptable NVE fluctuation ratio
  sqrt(<ΔE_tot²>/<ΔE_kin²>) < 10%; published force fields gave 14-152%
  untapered **[FW19]**. Tapered bond orders fix it without refitting **[FW19]**;
  lone-pair, SBO2 and torsion-angle discontinuities in **[FW20]**.
- LAMMPS has no bond-order tapering. Cutoffs (`thb_cutoff`, `hbond_cutoff`,
  the ffield bond-order cutoff) are effectively part of the force field
  **[FW19]**: fit and deploy with the same values.
- Inner wall: E_core = ecore·exp[acore(1 − r/rcore)], active when rcore2 and
  acore2 > 0.01 (LAMMPS source); vdW shielding is used when gamma_w > 0.5.

## 9. Validation and attainable accuracy

- Beyond training loss: held-out set with early stopping **[Sh19, P21]**;
  NVT/NVE MD at the temperatures of use (energy conservation, collapse,
  over-coordination, species) **[C08, FW19, NB19]**; DFT on MD snapshots
  **[Du24]**; comparison with the starting force field on the same data.
- Overfitting signature: low training error but unphysical MD **[Sh19]**.
- Realistic accuracy: ReaxFF MAE 19.6 kcal/mol vs CCSD(T) on phenolic
  pyrolysis bond/reaction energies and barriers (B3LYP 3.8, DFTB 13-17)
  **[Ba13]** *(abstract)*; Dumortier et al. found a 1.25 kcal/mol tolerance
  unattainable and reached 8.3 kcal/mol validation RMSE on adsorption
  energies **[Du24]**. No verified source supports a generic "2-5 kcal/mol";
  do not promise it.

## 10. Iterative refinement

- Training on MD-sampled DFT frames is established **[DA24, Du24]**, and the
  MD -> DFT -> refit loop is standard for neural force fields (seeded with
  DFT-relabeled ReaxFF-MD frames in [Yo21]). No peer-reviewed ReaxFF work
  found (Oct 2026) runs it with uncertainty-driven selection; this toolkit's
  `refine-select` uses committee disagreement (justified by [Sh19]'s advice
  to add data where near-optimal fits disagree) plus short-contact
  extrapolation. Say so when you describe it.

## Tools (verified)

JAX-ReaxFF (github.com/cagrikymk/JAX-ReaxFF, GPL-3.0; classic
trainset.in/geo/params/ffield), I-ReaxFF (github.com/fenggo/I-ReaxFF,
AGPL-3.0 per LICENSE), GloMPO (GPL-3.0), INDEEDopt (MIT), ParAMS
(commercial, SCM), GARFfield (academic registration), PuReMD (GPL-3.0).
The public ogolem repository has no ReaxFF backend.
