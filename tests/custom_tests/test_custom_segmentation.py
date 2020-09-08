import pytest

TEST_CASES = [
    (
        "LSTM networks, which we preview in Sec. 2, have been successfully",
        ["LSTM networks, which we preview in Sec. 2, have been successfully"],
    ),
    (
        "When the tree is simply a chain, both Eqs. 2–8 and Eqs. 9–14 reduce to the standard LSTM transitions, Eqs. 1.",
        [
            "When the tree is simply a chain, both Eqs. 2–8 and Eqs. 9–14 reduce to the standard LSTM transitions, Eqs. 1."
        ],
    ),
    (
        "We used fluorescence time-lapse microscopy (Fig. 1D; fig. S1 and movies S1 and S2) and computational",
        [
            "We used fluorescence time-lapse microscopy (Fig. 1D; fig. S1 and movies S1 and S2) and computational"
        ],
    ),
    (
        "Hill functions indeed fit the data well (Fig. 3A and Table 1).",
        ["Hill functions indeed fit the data well (Fig. 3A and Table 1)."],
    ),
    (
        "In order to produce sentence representations that fully capture the semantics of natural language, order-insensitive models are insufficient due to their inability to account for differences in meaning as a result of differences in word order or syntactic structure (e.g., “cats climb trees” vs. “trees climb cats”).",
        [
            "In order to produce sentence representations that fully capture the semantics of natural language, order-insensitive models are insufficient due to their inability to account for differences in meaning as a result of differences in word order or syntactic structure (e.g., “cats climb trees” vs. “trees climb cats”)."
        ],
    ),
    (
        "There is an average exact sparsity (fraction of zeros) of the hidden layers of 83.40% on MNIST and 72.00% on CIFAR10. Figure 3 (left) provides a better understanding of the influence of sparsity.",
        [
            "There is an average exact sparsity (fraction of zeros) of the hidden layers of 83.40% on MNIST and 72.00% on CIFAR10.",
            "Figure 3 (left) provides a better understanding of the influence of sparsity.",
        ],
    ),
    (
        "Sparsity has become a concept of interest, not only in computational neuroscience and machine learning but also in statistics and signal processing (Candes and Tao, 2005). It was first introduced in computational neuroscience in the context of sparse coding in the visual system (Olshausen and Field, 1997).",
        [
            "Sparsity has become a concept of interest, not only in computational neuroscience and machine learning but also in statistics and signal processing (Candes and Tao, 2005).",
            "It was first introduced in computational neuroscience in the context of sparse coding in the visual system (Olshausen and Field, 1997).",
        ],
    ),
    (
        "1) The first item. 2) The second item.",
        ["1) The first item.", "2) The second item."],
    ),
    (
        "two of these stages (in areas V1 and V2 of visual cortex) (Lee et al., 2008), and that they",
        [
            "two of these stages (in areas V1 and V2 of visual cortex) (Lee et al., 2008), and that they"
        ],
    ),
    pytest.param(
        "all neu-\nrons fire at", ["all neu-\nrons fire at"], marks=pytest.mark.xfail
    ),
    (
        "the support of the Defense Advanced Resarch Projects Agency (DARPA) Deep Exploration and Filtering of Text (DEFT) Program under Air Force Research Laboratory (AFRL) contract",
        [
            "the support of the Defense Advanced Resarch Projects Agency (DARPA) Deep Exploration and Filtering of Text (DEFT) Program under Air Force Research Laboratory (AFRL) contract"
        ],
    ),
    (
        "While proprietary environments such as Microsoft Robotics Studio [9] and Webots [10] have many commendable attributes, we feel there is no substitute for a fully open platform.",
        [
            "While proprietary environments such as Microsoft Robotics Studio [9] and Webots [10] have many commendable attributes, we feel there is no substitute for a fully open platform."
        ],
    ),
    (
        "We first produce sentence representations hL and hR for each sentence in the pair using a Tree-LSTM model over each sentence’s parse tree.",
        [
            "We first produce sentence representations hL and hR for each sentence in the pair using a Tree-LSTM model over each sentence’s parse tree."
        ],
    ),
    (
        "LSTM networks, which we review in Sec. 2, have been successfully applied to a variety of sequence modeling and prediction tasks, notably machine translation (Bahdanau et al., 2014; Sutskever et al., 2014), speech recognition (Graves et al., 2013), image caption generation (Vinyals et al., 2014), and program execution (Zaremba and Sutskever, 2014).",
        [
            "LSTM networks, which we review in Sec. 2, have been successfully applied to a variety of sequence modeling and prediction tasks, notably machine translation (Bahdanau et al., 2014; Sutskever et al., 2014), speech recognition (Graves et al., 2013), image caption generation (Vinyals et al., 2014), and program execution (Zaremba and Sutskever, 2014)."
        ],
    ),
    (
        "1 Introduction\n\nMost models for distributed representations of phrases and sentences—that is, models where realvalued vectors are used to represent meaning—fall into one of three classes: bag-of-words models, sequence models, and tree-structured models.",
        [
            "1 Introduction\n\n",
            "Most models for distributed representations of phrases and sentences—that is, models where realvalued vectors are used to represent meaning—fall into one of three classes: bag-of-words models, sequence models, and tree-structured models.",
        ],
    ),
    (
        "In this section, we will elaborate these philosophies and shows how they influenced the design and implementation of ROS.\n\nA. Peer-to-Peer\n\nA system built using ROS consists of a number of processes, potentially on a number of different",
        [
            "In this section, we will elaborate these philosophies and shows how they influenced the design and implementation of ROS.\n\n",
            "A. Peer-to-Peer\n\n",
            "A system built using ROS consists of a number of processes, potentially on a number of different",
        ],
    ),
    (
        "\n\n2 Long Short-Term Memory Networks\n\n\n\n2.1 Overview\n\nRecurrent neural networks (RNNs) are able to process input sequences of arbitrary length via the recursive application of a transition function on a hidden state vector ht.",
        [
            "\n\n2 Long Short-Term Memory Networks\n\n\n\n",
            "2.1 Overview\n\n",
            "Recurrent neural networks (RNNs) are able to process input sequences of arbitrary length via the recursive application of a transition function on a hidden state vector ht.",
        ],
    ),
    (
        "In order to address all three aspects, it is necessary to observe gene regulation in individual cells over time. Therefore, we built Bl-cascade[ strains of Escherichia coli, containing the l repressor and a downstream gene, such that both the amount of the repressor protein and the rate of expression of its target gene could be monitored simultaneously in individual cells (Fig. 1B). These strains incorporate a yellow fluorescent repressor fusion protein (cI-yfp) and a chromosomally integrated target promoter (P R ) controlling cyan fluorescent protein (cfp).",
        [
            "In order to address all three aspects, it is necessary to observe gene regulation in individual cells over time.",
            "Therefore, we built Bl-cascade[ strains of Escherichia coli, containing the l repressor and a downstream gene, such that both the amount of the repressor protein and the rate of expression of its target gene could be monitored simultaneously in individual cells (Fig. 1B).",
            "These strains incorporate a yellow fluorescent repressor fusion protein (cI-yfp) and a chromosomally integrated target promoter (P R ) controlling cyan fluorescent protein (cfp).",
        ],
    ),
    (
        "This is a sentence. (This is an interjected sentence.) This is also a sentence.",
        [
            "This is a sentence.",
            "(This is an interjected sentence.)",
            "This is also a sentence.",
        ],
    ),
    (
        "Thus, we first compute EMC 3 's response time-i.e., the duration from the initial of a call (from/to a participant in the target region) to the time when the decision of task assignment is made; and then, based on the computed response time, we estimate EMC 3 maximum throughput [28]-i.e., the maximum number of mobile users allowed in the MCS system. EMC 3 algorithm is implemented with the Java SE platform and is running on a Java HotSpot(TM) 64-Bit Server VM; and the implementation details are given in Appendix, available in the online supplemental material.",
        [
            "Thus, we first compute EMC 3 's response time-i.e., the duration from the initial of a call (from/to a participant in the target region) to the time when the decision of task assignment is made; and then, based on the computed response time, we estimate EMC 3 maximum throughput [28]-i.e., the maximum number of mobile users allowed in the MCS system.",
            "EMC 3 algorithm is implemented with the Java SE platform and is running on a Java HotSpot(TM) 64-Bit Server VM; and the implementation details are given in Appendix, available in the online supplemental material.",
        ],
    ),
    (
        "Random walk models (Skellam, 1951;Turchin, 1998) received a lot of attention and were then extended to several more mathematically and statistically sophisticated approaches to interpret movement data such as State-Space Models (SSM) (Jonsen et al., 2003(Jonsen et al., , 2005 and Brownian Bridge Movement Model (BBMM) (Horne et al., 2007). Nevertheless, these models require heavy computational resources (Patterson et al., 2008) and unrealistic structural a priori hypotheses about movement, such as homogeneous movement behavior. A fundamental property of animal movements is behavioral heterogeneity (Gurarie et al., 2009) and these models poorly performed in highlighting behavioral changes in animal movements through space and time (Kranstauber et al., 2012).",
        [
            "Random walk models (Skellam, 1951;Turchin, 1998) received a lot of attention and were then extended to several more mathematically and statistically sophisticated approaches to interpret movement data such as State-Space Models (SSM) (Jonsen et al., 2003(Jonsen et al., , 2005 and Brownian Bridge Movement Model (BBMM) (Horne et al., 2007).",
            "Nevertheless, these models require heavy computational resources (Patterson et al., 2008) and unrealistic structural a priori hypotheses about movement, such as homogeneous movement behavior.",
            "A fundamental property of animal movements is behavioral heterogeneity (Gurarie et al., 2009) and these models poorly performed in highlighting behavioral changes in animal movements through space and time (Kranstauber et al., 2012).",
        ],
    ),
    (". . .", [". . ."]),
    (
        "IF condition and goalCondition THEN action condition relates to the current state and goalCondition to the goal state. If variable bindings exist such that predicates in condition match with the current state, and predicates in goalCondition match with the goal state then the action may be performed. Note that the action's precondition as specified in the domain model must also be satisfied. Figure 5 presents an outline of the system. Each iteration starts with a population of policies (line(2)). Current L2Plan settings are such that the individuals comprising the (1) Create initial population (2) WHILE termination criterion false (3) Determine n% fittest polices (4) Perform local search on policies (5) Insert improved policies in new generation (6) WHILE new generation not full (7) SET Pol to empty policy (8) Select two parents (9) IF crossover (10) Perform crossover (11) Pol := fittest of parents & offspring (12) ELSE (13) Pol := fittest of parents (14) ENDIF (15) IF mutation (16) Perform mutation on Pol (17) ENDIF (18) Perform local search on Pol (19) Insert Pol in new generation (20) ENDWHILE (21) (5)). Note that the evaluation of policies is implied when the fittest policy or policies is/are required.",
        [
            "IF condition and goalCondition THEN action condition relates to the current state and goalCondition to the goal state.",
            "If variable bindings exist such that predicates in condition match with the current state, and predicates in goalCondition match with the goal state then the action may be performed.",
            "Note that the action's precondition as specified in the domain model must also be satisfied.",
            "Figure 5 presents an outline of the system.",
            "Each iteration starts with a population of policies (line(2)).",
            "Current L2Plan settings are such that the individuals comprising the (1) Create initial population (2) WHILE termination criterion false (3) Determine n% fittest polices (4) Perform local search on policies (5) Insert improved policies in new generation (6) WHILE new generation not full (7) SET Pol to empty policy (8) Select two parents (9) IF crossover (10) Perform crossover (11) Pol := fittest of parents & offspring (12) ELSE (13) Pol := fittest of parents (14) ENDIF (15) IF mutation (16) Perform mutation on Pol (17) ENDIF (18) Perform local search on Pol (19) Insert Pol in new generation (20) ENDWHILE (21) (5)).",
            "Note that the evaluation of policies is implied when the fittest policy or policies is/are required.",
        ],
    ),
    (
        "MCC summarizes these four quantities into one score and is regarded as a balanced measure; it takes values between -1 and 1, with higher values indicating better performance (see e.g. Baldi et al. (2000) for further details). Since the convergence threshold in the glasso algorithm is 10 −4 , we take entriesω ij in estimated precision matrices to be non-zero if |ω ij | > 10 −3 . Since cluster assignments can only be identified up to permutation, in all cases labels were permuted to maximize agreement with true cluster assignments before calculating these quantities. Figure 2 shows MCC plotted against per-cluster sample size n k and Supplementary Figure S1 shows corresponding plots for TPR and FPR. Due to selection of smaller tuning parameter values, BIC discovers fewer non-zeroes in the precision matrices than train/test, resulting in both fewer true positives and false positives. Under MCC, BIC, with either the γ = 1 mixture model (B1) or the non-mixture approach (Bh), leads to the best network reconstruction (except at small sample sizes with p = 25) and outperforms all other regimes at larger sample sizes.",
        [
            "MCC summarizes these four quantities into one score and is regarded as a balanced measure; it takes values between -1 and 1, with higher values indicating better performance (see e.g. Baldi et al. (2000) for further details).",
            "Since the convergence threshold in the glasso algorithm is 10 −4 , we take entriesω ij in estimated precision matrices to be non-zero if |ω ij | > 10 −3 .",
            "Since cluster assignments can only be identified up to permutation, in all cases labels were permuted to maximize agreement with true cluster assignments before calculating these quantities.",
            "Figure 2 shows MCC plotted against per-cluster sample size n k and Supplementary Figure S1 shows corresponding plots for TPR and FPR.",
            "Due to selection of smaller tuning parameter values, BIC discovers fewer non-zeroes in the precision matrices than train/test, resulting in both fewer true positives and false positives.",
            "Under MCC, BIC, with either the γ = 1 mixture model (B1) or the non-mixture approach (Bh), leads to the best network reconstruction (except at small sample sizes with p = 25) and outperforms all other regimes at larger sample sizes.",
        ],
    ),
    (
        'Societal impact measurements are mostly commissioned by governments which argue that measuring the impact on science little says about real-world benefits of research (Cohen et al., 2015). Nightingale and Scott (2007) summarize this argumentation in the following pointedly sentence: "Research that is highly cited or published in top journals may be good for the academic discipline but not for society" (p. 547). Governments are interested to know the importance of public-funded research (1) for the private and public sectors (e.g. health care), (2) to tackle societal challenges (e.g. climate change), and (3) for education and training of the next generations (ERiC, 2010;Grimson, 2014). The impact model of Cleary, Siegfried, Jackson, and Hunt (2013) additionally highlights the policy enactment of research, in which the impact on policies, laws, and regulations is of special interest. The current study seizes upon this additional issue by investigating a possible source for measuring policy enactment of research.',
        [
            "Societal impact measurements are mostly commissioned by governments which argue that measuring the impact on science little says about real-world benefits of research (Cohen et al., 2015).",
            'Nightingale and Scott (2007) summarize this argumentation in the following pointedly sentence: "Research that is highly cited or published in top journals may be good for the academic discipline but not for society" (p. 547).',
            "Governments are interested to know the importance of public-funded research (1) for the private and public sectors (e.g. health care), (2) to tackle societal challenges (e.g. climate change), and (3) for education and training of the next generations (ERiC, 2010;Grimson, 2014).",
            "The impact model of Cleary, Siegfried, Jackson, and Hunt (2013) additionally highlights the policy enactment of research, in which the impact on policies, laws, and regulations is of special interest.",
            "The current study seizes upon this additional issue by investigating a possible source for measuring policy enactment of research.",
        ],
    ),
    (
        "CONCLUSIONS: This study demonstrates that TF activation, occurring in mononuclear cells of cardiac transplant recipients, is inhibited by treatment with CsA. Inhibition of monocyte TF induction by CsA may contribute to its successful use in cardiac transplant medicine and might be useful in managing further settings of vascular pathology also known to involve TF expression and NF-kappaB activation.",
        [
            "CONCLUSIONS: This study demonstrates that TF activation, occurring in mononuclear cells of cardiac transplant recipients, is inhibited by treatment with CsA.",
            "Inhibition of monocyte TF induction by CsA may contribute to its successful use in cardiac transplant medicine and might be useful in managing further settings of vascular pathology also known to involve TF expression and NF-kappaB activation.",
        ],
    ),
    (
        "In contrast, anti-AIM mAb did not induce any change in the binding activity of NF-kappa B, a transcription factor whose activity is also regulated by protein kinase C. The increase in AP-1-binding activity was accompanied by the marked stimulation of the transcription of c-fos but not that of c-jun.",
        [
            "In contrast, anti-AIM mAb did not induce any change in the binding activity of NF-kappa B, a transcription factor whose activity is also regulated by protein kinase C. The increase in AP-1-binding activity was accompanied by the marked stimulation of the transcription of c-fos but not that of c-jun."
        ],
    ),
    (
        "A mutant Tax protein deficient in transactivation of genes by the nuclear factor (NF)-kappaB pathway was unable to induce transcriptional activity of IL-1alpha promoter-CAT constructs, but was rescued by exogenous provision of p65/p50 NF-kappaB. We found that two IL-1alpha kappaB-like sites (positions -1,065 to -1,056 and +646 to +655) specifically formed a complex with NF-kappaB-containing nuclear extract from MT-2 cells and that NF-kappaB bound with higher affinity to the 3' NF-kappaB binding site than to the 5' NF-kappaB site.",
        [
            "A mutant Tax protein deficient in transactivation of genes by the nuclear factor (NF)-kappaB pathway was unable to induce transcriptional activity of IL-1alpha promoter-CAT constructs, but was rescued by exogenous provision of p65/p50 NF-kappaB.",
            "We found that two IL-1alpha kappaB-like sites (positions -1,065 to -1,056 and +646 to +655) specifically formed a complex with NF-kappaB-containing nuclear extract from MT-2 cells and that NF-kappaB bound with higher affinity to the 3' NF-kappaB binding site than to the 5' NF-kappaB site.",
        ],
    ),
    pytest.param(
        "Protein kinase C inhibitor staurosporine, but not cyclic nucleotide-dependent protein kinase inhibitor HA-1004, also dramatically reduced constitutive levels of nuclear NF kappa B. Finally, TPA addition to monocytes infected with HIV-1 inhibited HIV-1 replication, as determined by reverse transcriptase assays, in a concentration-dependent manner.",
        [
            "Protein kinase C inhibitor staurosporine, but not cyclic nucleotide-dependent protein kinase inhibitor HA-1004, also dramatically reduced constitutive levels of nuclear NF kappa B.",
            "Finally, TPA addition to monocytes infected with HIV-1 inhibited HIV-1 replication, as determined by reverse transcriptase assays, in a concentration-dependent manner.",
        ],
        marks=pytest.mark.xfail,
    ),
    (
        "There are p50.c-rel heterodimers were also detected bound to this sequence at early time points (7-16 h; early), and both remained active at later time points (40 h; late) after activation.",
        [
            "There are p50.c-rel heterodimers were also detected bound to this sequence at early time points (7-16 h; early), and both remained active at later time points (40 h; late) after activation."
        ],
    ),
    (
        "This sentence mentions Eqs. 1-4 and should not be split.",
        ["This sentence mentions Eqs. 1-4 and should not be split."],
    ),
    (
        "This sentence ends with part an abbreviation that is part of a word material. It also has another sentence after it.",
        [
            "This sentence ends with part an abbreviation that is part of a word material.",
            "It also has another sentence after it.",
        ],
    ),
    (
        "It also has a sentence before it. This sentence mentions Eqs. 1-4 and should not be split. It also has another sentence after it.",
        [
            "It also has a sentence before it.",
            "This sentence mentions Eqs. 1-4 and should not be split.",
            "It also has another sentence after it.",
        ],
    ),
    (
        "This sentence is the last segment and ends with an abbreviation that is part of a word material.",
        [
            "This sentence is the last segment and ends with an abbreviation that is part of a word material."
        ],
    ),
    (
        "PDBu + iono induced equally high IL-2 levels in both groups and, when stimulated with plate-bound anti-CD3 monoclonal antibody (mAb), the IL-2 secretion by neonatal cells was undetectable and adult cells produced low amounts of IL-2 (mean 331 +/- 86 pg/ml).",
        [
            "PDBu + iono induced equally high IL-2 levels in both groups and, when stimulated with plate-bound anti-CD3 monoclonal antibody (mAb), the IL-2 secretion by neonatal cells was undetectable and adult cells produced low amounts of IL-2 (mean 331 +/- 86 pg/ml)."
        ],
    ),
    (
        "    This document starts with whitespaces. Next sentence.",
        ["    ", "This document starts with whitespaces.", "Next sentence."],
    ),
    pytest.param(
        "How about tomorrow?We can meet at eden garden.",
        ["How about tomorrow?", "We can meet at eden garden."],
        marks=pytest.mark.xfail,
    ),
]


@pytest.mark.parametrize("text,expected_sents", TEST_CASES)
def test_custom_segmentation(
    en_with_combined_rule_tokenizer_and_segmenter_fixture,
    remove_new_lines_fixture,
    text,
    expected_sents,
):
    doc = en_with_combined_rule_tokenizer_and_segmenter_fixture(text)
    sents = [s.text for s in doc.sents]
    assert sents == expected_sents


def test_segmenter(en_with_combined_rule_tokenizer_and_segmenter_fixture):
    # this text used to crash pysbd
    text = r"Then, (S\{ℓ 1 , ℓ 2 }) ∪ {v} is a smaller power dominating set than S, which is a contradiction. Now consider the case in which v ∈ V is incident to exactly two leaves, ℓ 1 and ℓ 2 , and suppose there is a minimum power dominating set S of G such that {v, ℓ 1 , ℓ 2 } ∩ S = ∅."
    doc = en_with_combined_rule_tokenizer_and_segmenter_fixture(text)
    # this is really just testing that we handle the case where pysbd crashes
    assert len(list(doc.sents)) > 0

    # this text used to crash pysbd
    text = r"Note that by definition of J, for i ∈ J, S i can be chosen such that S i \{v} is a set realizing γ P (T i ). By Lemma 3.8, for i ∈ I ′ , S i can be chosen such that v does not need to perform a force. Suppose first that |I| ≤ 1 and |J| ≥ 1; we claim that k i=1 (S i \{v}) is a power dominating set of T . To see why, note that for each i ∈ J, the set S i \{v} will force all of V (T i ) in T , including v. Then for i ∈ I ′ , all components T i can be forced by the sets S i \{v}, i ∈ I ′ , since v is colored but does not need to perform a force in those components. Finally, if there is a component T i * , i * ∈ I, v will have a single uncolored neighbor at this step of the forcing process (which is in T i * ), and it can force this neighbor; since v is a leaf in T i * , this is the same as dominating its neighbor. Thus, S i * \{v} can power dominate T i * after all other components are colored."
    doc = en_with_combined_rule_tokenizer_and_segmenter_fixture(text)
    # this is really just testing that we handle the case where pysbd crashes
    assert len(list(doc.sents)) > 0
