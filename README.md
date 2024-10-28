# LOFT: A 1 Million+ Token Long-Context Benchmark

This repository houses the resources for LOFT, the Long Context Frontiers benchmark, introduced in the research paper [Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?](https://arxiv.org/abs/2406.13121).
LOFT consists of 6 long-context task categories spanning retrieval, multi-hop compositional reasoning, and more, totaling 30+ datasets and 4 modalities.

We've provided links to download many of the text datasets in LOFT, evaluation code, and code to regenerate some of the datasets that we do not fully release.
We also provide an example prompt in `PROMPT_EXAMPLE.txt` showing how Corpus-in-Context (CiC) prompting can be done for the text retrieval task.

Install the dependencies in `requirements.txt` to use this repository.

**Future Releases**

* Multi-modal data.
* Task-specific prompts.

**Releases**

* [8/30/24]: Release of the evaluation code for ICL and some ICL and visual retrieval datasets.
* [6/29/24]: Release of the evaluation code for text tasks and code to regenerate some of the LOFT datasets.
* [6/20/24]: Initial release with links to download many of the LOFT text datasets.

## Dataset Creation via Infilling
For many of the datasets, we release the complete set of queries and corpus used in the LOFT paper via the links in the [Datasets](#datasets) table.
For a small subset, we require the user to first download using the links in the [Datasets](#datasets) table, then run `preprocess.py` which downloads the original dataset and infills the missing fields in the queries and corpus files.
The datasets that do require infilling have a ✅ under the `Infilling Needed?` column.

For example, FIQA for text retrieval requires infilling.
To infill the FIQA dataset, first download the ZIP file and unzip.
Then run:
```bash
python preprocess.py \
  --input_dir path/to/unzipped/fiqa \
  --dataset fiqa \
```

## Evaluation
To evaluate predictions:
```bash
python run_evaluation.py \
  --answer_file_path path/to/queries.jsonl \
  --pred_file_path path/to/preds.jsonl \
  --task_type <task_type>
```

We provide example queries and predictions files in  [evaluation/example_predictions/](evaluation/example_predictions/).
For example, to run evaluation on the RAG Natural Questions example predictions:
```bash
python run_evaluation.py \
  --answer_file_path evaluation/example_predictions/rag_nq/queries.jsonl \
  --pred_file_path evaluation/example_predictions/rag_nq/preds.jsonl \
  --task_type rag
```

The `task_type`'s are defined in [evaluation/__init__.py](evaluation/__init__.py).
Each `task_type` outputs many different metric scores.
To understand which `task_type` to use for each dataset and also to see the primary evaluation metric reported in the paper for each dataset, see the [Datasets](#datasets) table.

Evaluation expects a prediction file in a JSONLines format where each line has the following structure:

`{"qid": "test103", "num_turns": 1, "model_outputs": [["Spain"]]}`

* `qid`: QID of the prediction corresponding to an entry in the queries file.
* `num_turns`: Number of turns for the QID. This is 1 except for multi-turn datasets (TopiOCQA and SParC).
* `model_outputs`: The model predictions extracted as a list. We leave it to the user of LOFT to extract the model predictions into the right structure.

The required structure of the `model_outputs` field differs slightly for each `task_type`.
See [evaluation/example_predictions/](evaluation/example_predictions/) to understand how to format the predictions file.


## Datasets

| Task | Dataset | Description | Task Type | Primary Metric | Infilling Needed? | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Text Retrieval | [ArguAna](https://github.com/beir-cellar/beir) | Argument Retrieval | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/arguana.zip) |
| Text Retrieval | [FEVER](https://github.com/beir-cellar/beir) | Fact Checking | `retrieval` | `recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/fever.zip) |
| Text Retrieval | [FIQA](https://github.com/beir-cellar/beir) | Question Answering | `retrieval` | `recall@1` | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/fiqa.zip) |
| Text Retrieval | [MS MARCO](https://github.com/beir-cellar/beir) | Web Search | `retrieval` |`recall@1` | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/msmarco.zip) |
| Text Retrieval | [NQ](https://github.com/beir-cellar/beir) | Question Answering | `retrieval` |`recall@1` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/nq.zip) |
| Text Retrieval | [Quora](https://github.com/beir-cellar/beir) | Duplication Detection | `retrieval` |`recall@1` | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/quora.zip) |
| Text Retrieval | [SciFact](https://github.com/beir-cellar/beir) | Citation Prediction | `retrieval` |`recall@1`  | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/scifact.zip) |
| Text Retrieval | [Touché-2020](https://github.com/beir-cellar/beir) | Argument Retrieval | `retrieval` | `recall@1`  | ✅ | [Link](https://storage.googleapis.com/loft-bench/retrieval/webis_touche2020.zip) |
| Text Retrieval | [TopiOCQA](https://github.com/McGill-NLP/topiocqa) | Multi-turn QA | `retrieval` |`recall@1`  | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/topiocqa.zip) |
| Text Retrieval | [HotPotQA](https://github.com/beir-cellar/beir) | Multi-hop QA | `retrieval` | `mrecall@2` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/hotpotqa.zip) |
| Text Retrieval | [MuSiQue](https://allenai.org/data/musique) | Multi-hop QA | `retrieval` | `mrecall@5` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/musique.zip) |
| Text Retrieval | [QAMPARI](https://github.com/samsam3232/qampari) | Multi-target QA | `retrieval` |  `mrecall@5` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/qampari.zip) |
| Text Retrieval | [QUEST](https://github.com/google-research/language/tree/master/language/quest) | Multi-target QA | `retrieval` | `mrecall@3` | - | [Link](https://storage.googleapis.com/loft-bench/retrieval/quest.zip) |
| Visual Retrieval | [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) | Image Retrieval | `retrieval` | `recall@1` |✅ | Coming Soon |
| Visual Retrieval | [MS COCO](https://cocodataset.org) | Image Retrieval | `retrieval` | `recall@1` |✅ | Coming Soon |
| Visual Retrieval | [OVEN](https://github.com/open-vision-language/oven) | Image-text Retrieval | `retrieval` | `recall@1` | - | Coming Soon |
| Visual Retrieval | [MSR-VTT](https://cove.thecvf.com/datasets/839) | Video Retrieval | `retrieval` | `recall@1`| ✅ | [Link](https://storage.googleapis.com/loft-bench/mm/msrvtt.zip) |
| Audio Retrieval | [FLEURS-en](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | Coming Soon |
| Audio Retrieval | [FLEURS-es](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | Coming Soon |
| Audio Retrieval | [FLEURS-fr](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1`| - | Coming Soon |
| Audio Retrieval | [FLEURS-hi](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | Coming Soon |
| Audio Retrieval | [FLEURS-zh](https://huggingface.co/datasets/google/fleurs) | Audio Retrieval | `retrieval` | `recall@1` | - | Coming Soon |
| RAG | [NQ](https://github.com/beir-cellar/beir) | Question Answering | `rag` | `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/nq.zip) |
| RAG | [TopiOCQA](https://github.com/McGill-NLP/topiocqa) | Multi-turn QA | `rag` |  `subspan_em` | - | Coming Soon |
| RAG | [HotPotQA](https://github.com/beir-cellar/beir) | Multi-hop QA | `rag` |  `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/hotpotqa.zip) |
| RAG | [MuSiQue](https://allenai.org/data/musique) | Multi-hop QA | `rag` |  `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/musique.zip) |
| RAG | [QAMPARI](https://github.com/samsam3232/qampari) | Multi-target QA | `multi_value_rag` | `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/qampari.zip) |
| RAG | [QUEST](https://github.com/google-research/language/tree/master/language/quest) | Multi-target QA | `multi_value_rag` | `subspan_em` | - | [Link](https://storage.googleapis.com/loft-bench/rag/quest.zip) |
| SQL | [Spider](https://yale-lily.github.io/spider) | Single-turn SQL | `sql` | `exec_acc` | - | [Link](https://storage.googleapis.com/loft-bench/sql/spider.zip) |
| SQL | [SParC](https://yale-lily.github.io/sparc) | Multi-turn SQL | `sql` | `exec_acc` | - | [Link](https://storage.googleapis.com/loft-bench/sql/sparc.zip) |
| Many-Shot ICL | [BBH-date](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/date_understanding.zip) |
| Many-Shot ICL |[BBH-salient](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/salient_translation_error_detection.zip) |
| Many-Shot ICL |[BBH-tracking7](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/tracking_shuffled_objects_seven_objects.zip) |
| Many-Shot ICL |[BBH-web](https://github.com/suzgunmirac/BIG-Bench-Hard) | Multiple-choice QA | `icl` | `em` | - | [Link](https://storage.googleapis.com/loft-bench/icl/web_of_lies.zip) |
| Many-Shot ICL |[LIB-dialogue](https://github.com/TIGER-AI-Lab/LongICLBench) | Classification | - | - | ✅ | Coming Soon |

## Citing this work

```latex
@article{Lee2024LongContext,
  title={Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?},
  author={Jinhyuk Lee and Anthony Chen and Zhuyun Dai and Dheeru Dua and Devendra Singh Sachan and Michael Boratko and Yi Luan and Sébastien M. R. Arnold and Vincent Perot and Siddharth Dalmia and Hexiang Hu and Xudong Lin and Panupong Pasupat and Aida Amini and Jeremy R. Cole and Sebastian Riedel and Iftekhar Naim and Ming-Wei Chang and Kelvin Guu},
  journal={ArXiv},
  year={2024},
  volume={abs/2406.13121},
  url={https://arxiv.org/abs/2406.13121}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Individual tasks may be subject to copyright and licensing from their respective owners - please see individual download files for details.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
