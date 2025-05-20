# About this Notebook

View the notebook with plots and tables here: https://nbviewer.org/github/phibya/message-effect/blob/main/analysis.ipynb

This notebook demonstrates how to analyze the treatment effect of the test messages using different methods, specifically:
- Basic analysis: calculate the average treatment effect (ATE) for all messages and demographic segments without any model.
- Linear regression: use a linear regression model to analyze the treatment effect.
- Causal Forest: use a causal forest model to analyze the treatment effect and visualize the treatment effect by demographic segments.

The data used in this notebook is generated from the `generate_data.py` script. The data is a simulated survey data of 9262 respondents with the following columns:
- `message_id`: the id of the message (0 if control, 1-5 if treated)
- `message_content`: the content of the message
- `age`
- `gender`
- `race`
- `education`
- `sexual_orientation`
- `party`
- `ideology`
- `income`
- `religion`
- `marital_status`
- `residential_area`
- `turnout_score`: a score from 0 to 100 indicating the likelihood of the respondent to vote in the upcoming election.
- `vote_choice`: the vote choice of the respondent (Harris, Trump, or Undecided)

The data is generated with bias depending on the demographic.