# MLOps - A Comprehensive Guide

[Work in progress]

When it comes to machine learning, what most people care about is making the most of models and that usually means putting models into production. MLOPs standing for Machine Learning Operations involves set of processes that make it possible to design, train, evaluate, and to deploy models.

This page serves as a comprehensive guide to MLOps. It gives a brief introduction to MLOps and why it is an important area of study, relevant learning resources (such as courses, books, papers), tools, and various communities in MLOps. Here is a rough outline:

- [1. Introduction to MLOps](#1)
- [2. MLOps learning resources](#2)
    - [2.1 Courses](#2-1)
    - [2.2 Books](#2-2)
    - [2.3 Papers](#2-3)
    - [2.4 Blogs](#2-4)
    - [2.5 Active communities](#2-5)
- [3. MLOPs tools landscape](3)
- [4. Conclusion](#4)
- [5. References](#5)

Machine learning evolves faster than other other fields. Thus, new resources will evolve, new tools will come while existing tools will become irrelevant. This page will be updated continously(that's a hope) to ensure you get a practical blueprint for learning MLOps.

## 1. Introduction to MLOps

What is MLOps and why is everybody saying MLOps these days? As alluded to in the beginning of this tutorial, MLOps stands for Machine Learning Operations. It is a relatively new field(or nascent field according to [Shankar et al.](https://arxiv.org/abs/2209.09125))[1] that deals with operationalizing machine learning. MLOps roughly refers to a set of processes or methodologies designed to ensure reliable and efficient deployment of machine learning models in production.

MLOps workflow contains at least 6 stages which are discussed in brief below:

* **Data collection and cleaning**: This is the first and foremost task in any typical machine learning project. Data collection refers to collecting data from various sources. Data can be sourced from database, scraping internet, APIs, etc...Once the data is collected, the next step is usually cleaning it. This can involve labeling and other data wrangling activities.

* **Feature engineering:** Once data is collected and cleaned, you are not done with data yet. You may want to create new features, transform existing features, or extract new features from existing features. That's feature engineering is all about. It is a process of using domain knowledge to create new features or to transform existing features. Feature engineering is typically done after data cleaning and before building model.

* **Building and training models**: Model building is a step that follow data preprocessing. The kind of model and the tools you use roughly depends on the dataset and the problem. For instance, when working with structured data and performing discriminative tasks, your model maybe something like random forest or support vector machines or linear models. When performing image classification on large-scale dataset, your model maybe based on convolutional neural networks(CNNs). When doing sentiment analysis, your model maybe based on recurrent networks or finetuned [BERT](https://arxiv.org/abs/1810.04805). After designing the model, the next task is to train it. Model building and training are iterative processes and they involves lots of experimentations.

* **Model evaluation**: On the course of training, a model is evaluated on validation set(validation set can be a portion of training set that is not used in training a model) to compute performance metrics such as accuracy. Performance metrics depend on problem and dataset. For classification tasks, your performance metric can be accuracy. For regression, your metric can be mean-squarred error(MSE). Validation data are different to test data, validation data are used to evaluate the model during training while test data are used to evaluate the final model. When the model is good enough, the next step is to deploy it.

* **Model deployment**: Model deployment is the process of putting model into production to make it easily accessible to users, developers, or other applications. Model deployment is one of the last stages in machine learning life-cycle. Model can be deployed on cloud(via cloud services) or on edge-devices.

* **Model monitoring**: This step involves watching the performance of the deployed model overtime. Real-world is messy and many things can go wrong. It's important to monitor relevant pipelines(data or model related) and tracking key metrics so you can know when things break or when there is a bug causing mis-prediction.

Designing machine learning models is an iterative process. The ML life-cycle stages discussed above represent a rough overview of tasks involved in shipping ML models but it is not exhaustive and some stages are not straight-forward. While most of us spend time on building models, model is a tiny thing in entire ML lifecycle and there are other things that are not neccarily related to machine learning. There is nothing that depicts the complexity of MLOps that picture below.

![figure](https://github.com/Nyandwi/machine_learning_complete/raw/main/images/mlops-life.png)

## 2. MLOps Learning Resources

The following is a list of learning resources. The list highlights courses, books, papers, blogs, and active MLOps communities.

### 3.1 Courses

* Machine Learning Engineering for Production (MLOps) Specialization: This inarguable one of the best MLOps courses out there. It is taught by Andrew Ng., Laurence Moroney, and Robert Crowe. The first [course](https://www.youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK) of the specialization walk you through machine learning projects life-cycle while the rest courses focus on designing data pipelines, model pipelines and deploying models. The only caveat of the specialization is that the last 3 courses are all about [TensorFlow Extended](https://www.tensorflow.org/tfx). The entire course is available on [Coursera](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) and the first course [Introduction to Machine Learning in Production](https://www.youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK) of the specialization is available for free on YouTube.

* Full Stack Deep Learning(FSDL): FSD is inarguably the most practical MLOps course among all other MLops courses. Quoting the course website, "FSDL brings people together to learn and share best practices for the full stack: from problem selection, data management, and picking a GPU to web deployment, monitoring, and retraining." The course [website](https://fullstackdeeplearning.com/) provides more information about the course. The [course 2022 iteration](https://fullstackdeeplearning.com/course/2022/) is found here and course YouTube channel contains lectures(see [2022 playlist](https://www.youtube.com/playlist?list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur)).

* Effective MLOps: Model Development: This is a free course from awesome WandB that teaches how to build end-to-end machine learning pipelines. The course can be found [here](https://www.wandb.courses/courses/effective-mlops-model-development). Similar course for [CI/CD for Machine Learning (GitOps)](https://www.wandb.courses/courses/ci-cd-for-machine-learning) can be found here.

* MIT Introduction to Data-Centric AI(DCAI): This is the first-ever course on data-centric AI. While DCAI is relatively new field too, the practices done in DCAI are same as what's done in MLOps when working with data. The course materials can be found on the course [website](https://dcai.csail.mit.edu/) and lecture videos [here](https://www.youtube.com/watch?v=ayzOzZGHZy4&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5).

* Made With ML: Made With ML contains [resources](https://madewithml.com/) for learning ML foundations and MLOps, all through intuitive explanations, clean code and visualizations. Made With ML repository can be found [here](https://github.com/GokuMohandas/Made-With-ML). A dedicated MLOps repository for learning "how to combine machine learning with software engineering to develop, deploy and maintain production ML applications" can be found [here](https://github.com/GokuMohandas/mlops-course).


### 3.2 Books

* [Designing Machine Learning Systems by Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

* [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems by Martin Kleppmann](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/)

* [Introducing MlOps by Mark Treveil et al.](https://www.oreilly.com/library/view/introducing-mlops/9781492083283/)

* [Machine Learning Engineering by Andriy Burkov](http://www.mlebook.com/wiki/doku.php), free to read!

* [Beginning MLOps with MLFlow by Sridhar Alla ](https://link.springer.com/book/10.1007/978-1-4842-6549-9)

### 3.3 Papers


## 5. References