```mermaid

graph LR

    Entity_Linking_Core["Entity Linking Core"]

    Candidate_Generation["Candidate Generation"]

    Knowledge_Base_Management["Knowledge Base Management"]

    File_Caching_Utility["File Caching Utility"]

    UMLS_Semantic_Tree["UMLS Semantic Tree"]

    Entity_Linking_Core -- "uses" --> Candidate_Generation

    Candidate_Generation -- "accesses" --> Knowledge_Base_Management

    Candidate_Generation -- "loads data via" --> File_Caching_Utility

    Knowledge_Base_Management -- "loads data via" --> File_Caching_Utility

    Knowledge_Base_Management -- "builds hierarchy via" --> UMLS_Semantic_Tree

    UMLS_Semantic_Tree -- "loads data via" --> File_Caching_Utility

```

[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)



## Component Details



This architecture describes the core components involved in the entity linking process within scispacy. The main flow starts with the `Entity Linking Core` which orchestrates the overall linking. It utilizes the `Candidate Generation` component to find potential entities in text spans, which in turn relies on the `Knowledge Base Management` to access and retrieve information from various biomedical knowledge bases. Both `Candidate Generation` and `Knowledge Base Management` leverage the `File Caching Utility` for efficient data loading and access. Additionally, the `Knowledge Base Management` component interacts with the `UMLS Semantic Tree` to build and navigate the hierarchical structure of UMLS semantic types.



### Entity Linking Core

This component is responsible for the primary task of linking entities found in text to concepts within a knowledge base. It orchestrates the candidate generation and filtering process to identify the most relevant knowledge base entries for given text mentions.





**Related Classes/Methods**:



- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking.py#L9-L130" target="_blank" rel="noopener noreferrer">`scispacy.linking.EntityLinker` (9:130)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking.py#L69-L93" target="_blank" rel="noopener noreferrer">`scispacy.linking.EntityLinker.__init__` (69:93)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking.py#L95-L130" target="_blank" rel="noopener noreferrer">`scispacy.linking.EntityLinker.__call__` (95:130)</a>





### Candidate Generation

This component generates potential entity candidates for given text mentions. It utilizes TF-IDF vectorization and an approximate nearest neighbors (ANN) index to efficiently find similar concepts from the knowledge base. It also handles the creation and loading of these indices.





**Related Classes/Methods**:



- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py#L148-L361" target="_blank" rel="noopener noreferrer">`scispacy.candidate_generation.CandidateGenerator` (148:361)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py#L197-L235" target="_blank" rel="noopener noreferrer">`scispacy.candidate_generation.CandidateGenerator.__init__` (197:235)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py#L292-L361" target="_blank" rel="noopener noreferrer">`scispacy.candidate_generation.CandidateGenerator.__call__` (292:361)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py#L237-L290" target="_blank" rel="noopener noreferrer">`scispacy.candidate_generation.CandidateGenerator.nmslib_knn_with_zero_vectors` (237:290)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py#L364-L474" target="_blank" rel="noopener noreferrer">`scispacy.candidate_generation.create_tfidf_ann_index` (364:474)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/candidate_generation.py#L116-L145" target="_blank" rel="noopener noreferrer">`scispacy.candidate_generation.load_approximate_nearest_neighbours_index` (116:145)</a>





### Knowledge Base Management

This component handles the loading, representation, and access of various biomedical knowledge bases. It provides mappings between concept IDs, canonical names, and aliases, enabling efficient lookup of entity information. It includes general and specific knowledge base implementations for UMLS, MeSH, Gene Ontology, Human Phenotype Ontology, and RxNorm.





**Related Classes/Methods**:



- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L40-L76" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.KnowledgeBase` (40:76)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L52-L76" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.KnowledgeBase.__init__` (52:76)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L79-L89" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.UmlsKnowledgeBase` (79:89)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L80-L89" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.UmlsKnowledgeBase.__init__` (80:89)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L92-L97" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.Mesh` (92:97)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L93-L97" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.Mesh.__init__` (93:97)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L100-L105" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.GeneOntology` (100:105)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L101-L105" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.GeneOntology.__init__` (101:105)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L108-L113" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.HumanPhenotypeOntology` (108:113)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L109-L113" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.HumanPhenotypeOntology.__init__` (109:113)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L116-L121" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.RxNorm` (116:121)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/linking_utils.py#L117-L121" target="_blank" rel="noopener noreferrer">`scispacy.linking_utils.RxNorm.__init__` (117:121)</a>





### File Caching Utility

This utility component provides functionality for robust file access, including downloading and caching files from URLs or verifying local file paths. It ensures that necessary data files for knowledge bases and indices are readily available.





**Related Classes/Methods**:



- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/file_cache.py#L21-L50" target="_blank" rel="noopener noreferrer">`scispacy.file_cache.cached_path` (21:50)</a>





### UMLS Semantic Tree

This component is specifically designed to construct and represent the hierarchical structure of UMLS semantic types from a TSV file. It provides a tree-like view of the semantic relationships within the UMLS knowledge base, enabling efficient lookup and manipulation of semantic type hierarchies.





**Related Classes/Methods**:



- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/umls_semantic_type_tree.py#L81-L114" target="_blank" rel="noopener noreferrer">`scispacy.umls_semantic_type_tree.construct_umls_tree_from_tsv` (81:114)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/umls_semantic_type_tree.py#L105-L109" target="_blank" rel="noopener noreferrer">`scispacy.umls_semantic_type_tree.construct_umls_tree_from_tsv.attach_children` (105:109)</a>

- <a href="https://github.com/allenai/scispacy/blob/master/scispacy/umls_semantic_type_tree.py#L13-L78" target="_blank" rel="noopener noreferrer">`scispacy.umls_semantic_type_tree.UmlsSemanticTypeTree` (13:78)</a>









### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)