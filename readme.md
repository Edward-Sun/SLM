### Segmental Language Models

**Introduction**

A PyTorch Implementation of [Unsupervised Neural Word Segmentation for Chinese via Segmental Language Modeling](https://aclweb.org/anthology/D18-1531)

**Implemented features**

Models:

- Unsupervised Learning with Segmental Language Models
- Supervised Learning with Segmental Language Models

**Usage**

Chinese Corpus:

- *segmented.txt*: segmented data set for supervised training
- *unsegmented.txt*: unsegmented data set. You can use both this data set and *test.txt* for unsupervised training
- *test.txt*: unsegmented data set for evaluation
- *test_gold.txt*: gold segmented test data set

**Train**

For example, this command train an unsupervised SLM model on pku dataset with maximal segment length 4 and GPU 0.

```shell
bash run.sh train unsupervised pku 4 0
```

Check run.sh and argparse configuration at codes/run.py for more arguments and more details.

**Predict**

```shell
bash run.sh predict unsupervised pku 4 0
```

**Evaluation**

```shell
bash run.sh eval unsupervised pku 4
```

**Speed**

The Segmental Language Models usually take about 30 - 50 minutes to converge, which depends on the maximal segment length (2 - 4).

**Unsupervised results of the SLM model (Maximal Segment Length = k)**

| Dataset | PKU           | MSR           | AS            | CityU         |
| ------- | ------------- | ------------- | ------------- | ------------- |
| k = 2   | 0.797 (0.802) | 0.776 (0.785) | 0.794 (0.794) | 0.786 (0.782) |
| k = 3   | 0.803 (0.798) | 0.784 (0.794) | 0.800 (0.803) | 0.803 (0.805) |
| k = 4   | 0.797 (0.792) | 0.782 (0.790) | 0.798 (0.804) | 0.798 (0.797) |

Note that this is a re-implementation of the SLM model. Due to the differences in detailed settings, such as data loader setting, dropout rate and learning rate, the re-implementation performance is a little different from what is reported in the paper.

**Using the library**

The python library is organized around 4 objects:

- InputDataset (dataloader.py): prepare data stream for training and evaluation
- CWSTokenizer (tokenization.py): work along with InputDataset for data pre-processing
- SegmentalLM (model.py): build the model and provide train/test API for SLM
- SLMConfig (model.py): manage configurations for SLM

The run.py file contains the main function, which parses arguments, reads data, initialize the model and provides the training loop.

**Citation**

If you use the codes, please cite the following [paper](https://aclweb.org/anthology/D18-1531):

```
@inproceedings{sun2018unsupervised,
  title={Unsupervised Neural Word Segmentation for Chinese via Segmental Language Modeling},
  author={Sun, Zhiqing and Deng, Zhi-Hong},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={4915--4920},
  year={2018}
}
```