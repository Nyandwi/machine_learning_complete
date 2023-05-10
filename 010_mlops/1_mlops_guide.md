# MLOps - A Comprehensive Guide

When it comes to machine learning, what most people care about is making the most of models and that usually means putting models into production. MLOPs standing for Machine Learning Operations involves set of processes that make it possible to design, train, evaluate, and to deploy models.

This page serves as a comprehensive guide to MLOps. It gives a brief introduction to MLOps and why it is an important area of study, relevant learning resources (such as courses, books, papers), tools, and various communities in MLOps. Here is a rough outline:

- Introduction to MLOps
- MLOps learning resources
    - Courses
    - Books
    - Papers
    - Blogs, people and communities
- MLOPs tools landscape
- Conclusion

Machine learning evolves faster than other other fields. Thus, new resources will evolve, new tools will come while existing tools will become irrelevant. This page will be updated continously(my hope) to ensure you get a practical blueprint for learning MLOps.

## 1. Introduction to MLOps

What is MLOps and why is everybody saying MLOps these days? As alluded to in the beginning of this tutorial, MLOps stands for Machine Learning Operations. It is a relatively new field(or nascent field according to [Shankar et al.](https://arxiv.org/abs/2209.09125))that deals with operationalizing machine learning. MLOps roughly refers to a set of processes or methodologies designed to ensure reliable and efficient deployment of machine learning models in production.

MLOps workflow contains at least 6 stages which are discussed in brief below:

* **Data collection and cleaning**: This is the first and foremost task in any typical machine learning project. Data collection refers to collecting data from various sources. Data can be sourced from database, scraping internet, APIs, etc...Once the data is collected, the next step is usually cleaning it. This can involve labeling and other data wrangling activities.

* **Feature engineering:** Once data is collected and cleaned, you are not done with data yet. You may want to create new features, transform existing features, or extract new features from existing features. That's what feature engineering is all about. It is a process of using domain knowledge to create new features or to transform existing features. Feature engineering is typically done after data cleaning and before building model.

* **Building and training models**: Model building is a step that follow data preprocessing. The kind of model and the tools you use roughly depends on the dataset and the problem. For instance, when working with structured data and performing discriminative tasks, your model maybe something like random forest or support vector machines or linear models. When performing image classification on large-scale dataset, your model maybe based on convolutional neural networks(CNNs). When doing sentiment analysis, your model maybe based on recurrent networks or finetuned [BERT](https://arxiv.org/abs/1810.04805). After designing the model, the next task is to train it. Model building and training are iterative processes and they involves lots of experimentations.

* **Model evaluation**: On the course of training, a model is evaluated on validation set(validation set can be a portion of training set that is not used in training a model) to compute performance metrics such as accuracy. Performance metrics depend on problem and dataset. For classification tasks, your performance metric can be accuracy. For regression, your metric can be mean-squared error(MSE). Validation data are different to test data. Validation data are used to evaluate the model during training while test data are used to evaluate the final model after training. When the model is good enough, the next step is to deploy it.

* **Model deployment**: Model deployment is the process of putting model into production to make it easily accessible to users, developers, or other applications. Model deployment is one of the last stages in machine learning life-cycle. Model can be deployed on cloud(via cloud services) or on edge-devices.

* **Model monitoring**: This step involves watching the performance of the deployed model overtime. Real-world is messy and many things can go wrong. Overtime, data can change(data drift) or a model can decay(concept drift). It's important to monitor relevant pipelines(data or model related) and tracking key metrics so you can know when things break or when there is a bug causing mis-prediction.

Designing machine learning models is an iterative process. The ML life-cycle stages discussed above represent a rough overview of tasks involved in shipping ML models but it is not exhaustive and some stages are not straight-forward. While most of us spend time on building models, model is a tiny thing in entire ML lifecycle and there are other things that are not neccarily related to machine learning. There is nothing that depicts the complexity of MLOps than the picture below.

![figure](https://github.com/Nyandwi/machine_learning_complete/raw/main/images/mlops-life.png)

## 2. MLOps Learning Resources

The following is a list of learning resources. The list highlights courses, books, papers, blogs, and active MLOps communities.

### 2.1 Courses

* Machine Learning Engineering for Production (MLOps) Specialization: This inarguable one of the best MLOps courses out there. It is taught by Andrew Ng., Laurence Moroney, and Robert Crowe. The first [course](https://www.youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK) of the specialization walks you through machine learning projects life-cycle while the rest courses focus on designing data pipelines, model pipelines and deploying models. The only caveat(might not a caveat if you are working with TF ecosystem) of the specialization is that the last 3 courses are all about [TensorFlow Extended](https://www.tensorflow.org/tfx). The entire course is available on [Coursera](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) and the first course [Introduction to Machine Learning in Production](https://www.youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK) of the specialization is available for free on YouTube.

* Full Stack Deep Learning(FSDL): FSD is inarguably the most practical MLOps course among all other MLOps courses. Quoting the course website, "FSDL brings people together to learn and share best practices for the full stack: from problem selection, data management, and picking a GPU to web deployment, monitoring, and retraining." The course [website](https://fullstackdeeplearning.com/) provides more information about the course. The course 2022 iteration is found [here](https://fullstackdeeplearning.com/course/2022/) and course YouTube channel contains lectures(see [2022 playlist](https://www.youtube.com/playlist?list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur)).

* CS 329S: Machine Learning Systems Design: CS 329S provides an iterative framework for developing, deploying reliable and scalable machine learning systems. The course covers a wide range of topics such as data management, data engineering, feature engineering, model selection approaches, training, scaling, deploying and monitoring ML systems, and human side of ML projects. The [lecture notes and slides](https://stanford-cs329s.github.io/syllabus.html) are publicly available.

* Effective MLOps: Model Development: This is a free course from awesome WandB that teaches how to build end-to-end machine learning pipelines. The course can be found [here](https://www.wandb.courses/courses/effective-mlops-model-development). Similar course for CI/CD for Machine Learning (GitOps) can be found [here](https://www.wandb.courses/courses/ci-cd-for-machine-learning).

* MIT Introduction to Data-Centric AI(DCAI): This is the first-ever course on data-centric AI. While DCAI is relatively new field too, the practices done in DCAI are same as what's done in MLOps when working with data. The course materials can be found on the course [website](https://dcai.csail.mit.edu/) and lecture videos [here](https://www.youtube.com/watch?v=ayzOzZGHZy4&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5).

* Made With ML: Made With ML contains [resources](https://madewithml.com/) for learning ML foundations and MLOps, all through intuitive explanations, clean code and visualizations. Made With ML repository can be found [here](https://github.com/GokuMohandas/Made-With-ML). A dedicated MLOps repository for learning "how to combine machine learning with software engineering to develop, deploy and maintain production ML applications" can be found [here](https://github.com/GokuMohandas/mlops-course).


### 2.2 Books

The following are a few popular books in MLOps world. These books covers almost anything you'd want to know about MLOps. The books are not listed in any order but if you are to pick one book, take Chip Huyen book on designing ML systems or MLE by Burkov or Kleppmann book.

* [Designing Machine Learning Systems by Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

* [Machine Learning Engineering by Andriy Burkov](http://www.mlebook.com/wiki/doku.php), free to read!

* [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems by Martin Kleppmann](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/)

* [Introducing MLOps by Mark Treveil et al.](https://www.oreilly.com/library/view/introducing-mlops/9781492083283/)

* [Beginning MLOps with MLFlow by Sridhar Alla ](https://link.springer.com/book/10.1007/978-1-4842-6549-9)

* [Rules of Machine Learning:  Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml), free to read!

### 2.3 Papers

As alluded to in the beginning of this guide, MLOps is a new field in both industry and completely new field in academia and that means there are not many academic literatures on the topic. Below, we list few papers that are worth reading. If there is a paper we missed, feel free to contact me on [Twitter](https://twitter.com/Jeande_d).

* A Meta-Summary of Challenges in Building Products with ML Components – Collecting Experiences from 4758+ Practitioners: [ArXiv](https://arxiv.org/abs/2304.00078) | Mar 2023

* Operationalizing Machine Learning: An Interview Study: [ArXiv](https://arxiv.org/abs/2209.09125) | [Tweet](https://twitter.com/Jeande_d/status/1573397762170757120) | [Video](https://www.youtube.com/watch?v=LdMydLBDgEQ), [Transcript](https://wandb.ai/wandb_fc/gradient-dissent/reports/Shreya-Shankar-Operationalizing-Machine-Learning--VmlldzozNjg4MzUz) | Sep 2022

* Machine Learning Operations (MLOps): Overview, Definition, and Architecture: [ArXiv](https://arxiv.org/abs/2205.02302) | May 2022

 * Adoption and Effects of Software Engineering Best Practices in Machine Learning: [ArXiv](https://arxiv.org/abs/2007.14130) | Jul 2020


* Hidden Technical Debt in Machine Learning Systems: [Paper](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) | 2015

### 2.4 Blogs, People, and MLOps Communities.

There are way too many blogs about MLOps. In this section, instead of listing blog titles, we provide a few list of popular blogs(and authors) that write about MLOps or related topics. Related communities are also included.

* [Chip Huyen Blog](https://huyenchip.com/blog/): Chip Huyen writes a lot about designing machine learning systems and putting them in production. Her popular blogs related to MLOps are [Machine Learning Tools Landscape v2 (+84 new tools)](https://huyenchip.com/2020/12/30/mlops-v2.html), [Real-time machine learning: challenges and solutions](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html), and [Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html).

* [Eugene Yan](https://eugeneyan.com/): Eugene writes about designing and operating machine learning systems. His popular writings about ML in production can be found [here](https://eugeneyan.com/tag/production/). He also maintains a list of papers and tech blogs about [applied ML](maintain) and is creator of [applying ML](https://applyingml.com/).

* [Lj Miranda Notebook](https://ljvmiranda921.github.io/notebook/): Miranda documents his experiments and share study notes on different topics. His popular blogs about MLOps are [How to improve software engineering skills as a researcher](https://ljvmiranda921.github.io/notebook/2020/11/15/data-science-swe/), Navigating the MLOps tooling landscape ([Part 1: The Lifecycle](https://ljvmiranda921.github.io/notebook/2021/05/10/navigating-the-mlops-landscape/), [Part 2: The Ecosystem](https://ljvmiranda921.github.io/notebook/2021/05/15/navigating-the-mlops-landscape-part-2/), [Part 3: The Strategies](https://ljvmiranda921.github.io/notebook/2021/05/30/navigating-the-mlops-landscape-part-3/).

* [MLOps Community Blog](https://mlops.community/blog/): MLOps Community one of the best MLOps communities out there with exclusive blogs on the topic. A few examples of their blogs: [The Minimum Set of Must-Haves for MLOps](https://mlops.community/the-minimum-set-of-must-haves-for-mlops/), [MLOps is 98% Data Engineering](https://mlops.community/mlops-is-mostly-data-engineering/), [A Practitioner’s Guide to Monitoring Machine Learning Applications](https://mlops.community/guide-to-monitoring-machine-learning-applications/). In addition to blogs, MLOps community has a great [podcast](https://podcasts.apple.com/us/podcast/mlops-community/id1505372978).

* [Software Engineering for Machine Learning](https://se-ml.github.io/): SE ML collects, validates and shares machine learning engineering best practices. You can check their [Engineering best practices for Machine Learning](https://se-ml.github.io/practices/) and [The 2020 State of Engineering Practices for Machine Learning](https://se-ml.github.io/report2020).

 * [Awesome MLOps](https://github.com/kelvins/awesome-mlops) provides a curated list of awesome MLOps tools.

## 3. MLOPs Tools Landscape

Fields that are mature tend to have standard tools that every developer can point to easily when you ask them. MLOps is not like that yet. There are way too many tools since everybody is trying to contribute and it's almost impossible to list every tool. We hope the tools will mature overtime. In the meantime, I think it's good to follow tools that people in MLOps recommend. You can check them in one of the blogs we mentioned such as [Machine Learning Tools Landscape v2 (+84 new tools)](https://huyenchip.com/2020/12/30/mlops-v2.html) and Navigating the MLOps tooling landscape ([Part 1: The Lifecycle](https://ljvmiranda921.github.io/notebook/2021/05/10/navigating-the-mlops-landscape/), [Part 2: The Ecosystem](https://ljvmiranda921.github.io/notebook/2021/05/15/navigating-the-mlops-landscape-part-2/), [Part 3: The Strategies](https://ljvmiranda921.github.io/notebook/2021/05/30/navigating-the-mlops-landscape-part-3/). Shankar et al. and Kreuzberger et al. also provide a list of MLOps tools in their papers [Operationalizing Machine Learning: An Interview Study](https://arxiv.org/abs/2209.09125) and [Machine Learning Operations (MLOps): Overview, Definition, and Architecture](https://arxiv.org/abs/2205.02302) respectively.

## 4. Conclusion

Thanks for checking this MLOps guide. MLOps is a huge and interdisciplinary field that combines best practices from machine learning, software engineering, and data engineering. We shared lots of resources but and we understand you can't take all. Taking one course can get you started. Reading one book can get you started. Doing one project can get you started. Reading one blog can help you learn something new.

If there is a course or paper or blog or book that you think should be added to this guide, feel free to reach out on [Twitter](https://twitter.com/Jeande_d).

************

This guide is a part of [Complete Machine Learning Package](https://nyandwi.com/machine_learning_complete/).