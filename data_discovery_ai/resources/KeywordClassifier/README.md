# Resources
This document explains the artifacts generated and used for the ML models supporting Data Discovery AI features. These include:
1. Keyword Classification: Provides keyword suggestions for metadata records that lack keyword information. The related resources are saved in the `KeywordClassifier` folder.
2. Data Delivery Mode: Assists with classifying data delivery modes used by the AODN portal filter. The related resources are saved in the `DataDeliveryModeFilter` folder.

## `resources/*/*.keras`
These files with suffix `.keras` indicate the pretrained models. The file name (without suffix) are used for selecting models, which are restrictly controlled within these options:

| Option | Purpose | Typical Use |
| ---- | ---- | ---- |
| `development` | Dedicated to active model development, testing, and iteration. | Building and refining new model versions, features, or datasets. |
| `experimental` | Supports exploratory work for new techniques or fine-tuning. | Experimenting with new architectures, features, or hyperparameter tuning. |
| `staging` | Prepares the model for production with real-use evaluations. | Conducting final testing in a production-like environment to verify stability and performance. |
| `production` | Deployment environment for live model usage in real-world scenarios. | Running and monitoring models in active use by API. |
| `benchmark` | Baseline model used to assess improvements or changes. | Comparing performance metrics against new models. |

The deployment of these versions should follow this workflow: `development` -> `experimental` -> `benchmark` -> `staging` -> `production`

### Acceptance Cretrials
#### Development Stage
- [ ] Model is developted.
- [ ] Model meets basic performance benchmarks and has stable training metrics.
- [ ] Model is sufficiently tuned for further experimentation.

#### Experimental Stage
- [ ] Model has consistent and improved performance metrics.
- [ ] Experiment logs are well-documented for reproducibility and understanding.

#### Benchmark Stage
- [ ] Model performance exceed benchmark model(the copy of the production model) on selected evaluation metrics.

#### Staging Stage
- [ ] Model is ready to go to production
- [ ] All integrations (i.e., API services) are validated and tested.

#### Production Stage
- [ ] Deploy the model with monitoring metrics, user feedbacks, and source data changes.
