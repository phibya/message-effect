import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

seed = 1
np.random.seed(seed)
random.seed(seed)
fake = Faker()
Faker.seed(seed)

# Number of respondents
n_respondents = 9262  # Steal from the Blue Rose report :D

# Generate messages
messages = [
    {"message_id": 0, "message_content": "CONTROL - No message shown"},
    {"message_id": 1,
     "message_content": "In 2025 when the next president takes office, four of the nine Supreme Court justices will be in their 70s and could retire or die. That means Donald Trump could ensure MAGA control of the Court for decades to come, which could have a huge impact on upcoming cases concerning abortion, marriage equality, gun safety, and the climate crisis."},
    {"message_id": 2,
     "message_content": "Donald Trump brags that he personally selected three of the Supreme Court justices that 'killed Roe v. Wade.' Now, one in three women live in states where they cannot access abortion care when they need it."},
    {"message_id": 3,
     "message_content": "In 2025, four of the nine Supreme Court justices will be in their 70s and could retire or die. That means Donald Trump could ensure MAGA control of the Court for decades to come, threatening our access to abortion care and further undermining Americans' access to life-saving health care and treatments like IVF."},
    {"message_id": 4,
     "message_content": "The Supreme Court has undermined our democracy and our freedom to vote. They gutted the Voting Rights Act and let MAGA state lawmakers create barriers to the ballot box, particularly for communities of color and young voters."},
    {"message_id": 5,
     "message_content": "Again and again, MAGA justices on the Supreme Court are siding with big corporations over workers and our environment. They've weakened critical environmental rules protecting our air and water."}
]

# Demographics
age_ranges = list(range(18, 95))  # nobody < 18 can vote anyway
age_props = np.array([0.03 if 18 <= a <= 29 else
                      0.035 if 30 <= a <= 44 else
                      0.025 if 45 <= a <= 64 else
                      0.01 for a in age_ranges])
age_props = age_props / sum(age_props)

genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
gender_props = np.array([0.48, 0.49, 0.02, 0.01])

races = ["White", "Black", "Hispanic", "Asian", "Native American", "Multiple races", "Other", "Prefer not to say"]
race_props = np.array([0.6, 0.11, 0.18, 0.06, 0.01, 0.02, 0.01, 0.01])

education_levels = ["Less than high school", "High school graduate", "Some college", "Associate's degree",
                    "Bachelor's degree", "Graduate degree", "Prefer not to say"]
education_props = np.array([0.05, 0.25, 0.2, 0.1, 0.25, 0.14, 0.01])

sexual_orientations = ["Heterosexual", "Gay or Lesbian", "Bisexual", "Other", "Prefer not to say"]
sexual_orientation_props = np.array([0.9, 0.03, 0.04, 0.01, 0.02])

party_affiliations = ["Strong Democrat", "Moderate Democrat", "Lean Democrat", "Independent",
                      "Lean Republican", "Moderate Republican", "Strong Republican", "Other", "Prefer not to say"]
party_props = np.array([0.15, 0.15, 0.08, 0.1, 0.09, 0.17, 0.2, 0.04, 0.02])

ideologies = ["Very liberal", "Liberal", "Slightly liberal", "Moderate", "Slightly conservative",
              "Conservative", "Very conservative", "Prefer not to say"]
ideology_props = None  # None for now, to be determined based on party affiliation

income_ranges = ["Under $25,000", "$25,000-$49,999", "$50,000-$74,999", "$75,000-$99,999",
                 "$100,000-$149,999", "$150,000 or more", "Prefer not to say"]
income_props = np.array([0.15, 0.25, 0.2, 0.15, 0.15, 0.08, 0.02])

religion_affiliations = ["Protestant", "Catholic", "Mormon", "Orthodox Christian", "Jewish", "Muslim",
                         "Buddhist", "Hindu", "Atheist", "Agnostic", "Nothing in particular", "Other",
                         "Prefer not to say"]
religion_props = np.array([0.2, 0.2, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.08, 0.1, 0.3, 0.02, 0.02])

marital_statuses = ["Single, never married", "Married", "Divorced", "Separated", "Widowed", "Living with partner",
                    "Prefer not to say"]
marital_status_props = np.array([0.25, 0.45, 0.1, 0.03, 0.07, 0.09, 0.01])

residential_areas = ["Urban", "Suburban", "Rural", "Prefer not to say"]
residential_area_props = np.array([0.3, 0.5, 0.19, 0.01])

vote_choices = ["Harris", "Trump", "Undecided"]  # Added "Undecided" as a realistic option
vote_choice_props = np.array([0.5, 0.5, 0.05])

# Probability to create bias in the data
party_vote_probs = {
    "Strong Democrat": 0.95, "Moderate Democrat": 0.85, "Lean Democrat": 0.75,
    "Independent": 0.48, "Lean Republican": 0.25, "Moderate Republican": 0.15, "Strong Republican": 0.05,
    "Other": 0.5, "Prefer not to say": 0.5
}

ideology_vote_probs = {
    "Very liberal": 0.9, "Liberal": 0.85, "Slightly liberal": 0.75, "Moderate": 0.55,
    "Slightly conservative": 0.35, "Conservative": 0.15, "Very conservative": 0.05,
    "Prefer not to say": 0.5
}

race_vote_probs = {
    "White": 0.48, "Black": 0.85, "Hispanic": 0.65, "Asian": 0.65,
    "Native American": 0.60, "Multiple races": 0.65, "Other": 0.55,
    "Prefer not to say": 0.5
}

age_brackets = {
    (18, 29): 0.65,
    (30, 44): 0.58,
    (45, 64): 0.48,
    (65, 94): 0.45
}

education_vote_probs = {
    "Less than high school": 0.45, "High school graduate": 0.45, "Some college": 0.48,
    "Associate's degree": 0.50, "Bachelor's degree": 0.60, "Graduate degree": 0.65,
    "Prefer not to say": 0.5
}

# Treatment effects for different messages
message_effects = {
    0: 0.0,  # Control group (no effect)
    # other groups, steal from the report of the Blue Rose study
    1: 0.031,
    2: 0.028,
    3: 0.027,
    4: 0.024,
    5: 0.024
}

# Higher treatment effects for specific demographic groups based on the report
demo_message_effects = {
    "age_under_35": 0.04,
    "race_Black": 0.043,
    "race_Hispanic": 0.044,
    "ideology_Moderate": 0.043,
    "low_turnout": 0.055  # Additional effect for low turnout voters, will need to simulate this
}

data = []

for i in range(n_respondents):
    person_id = i + 1000

    age = np.random.choice(age_ranges, p=age_props)
    gender = np.random.choice(genders, p=gender_props)
    race = np.random.choice(races, p=race_props)
    education = np.random.choice(education_levels, p=education_props)
    sexual_orientation = np.random.choice(sexual_orientations, p=sexual_orientation_props)
    income = np.random.choice(income_ranges, p=income_props)
    religion = np.random.choice(religion_affiliations, p=religion_props)
    marital_status = np.random.choice(marital_statuses, p=marital_status_props)
    residential_area = np.random.choice(residential_areas, p=residential_area_props)

    party = np.random.choice(party_affiliations, p=party_props)

    if party in ["Strong Democrat", "Moderate Democrat"]:
        ideology_probs = [0.3, 0.4, 0.15, 0.1, 0.03, 0.01, 0.0, 0.01]
    elif party in ["Lean Democrat", "Independent"]:
        ideology_probs = [0.05, 0.15, 0.25, 0.35, 0.15, 0.03, 0.01, 0.01]
    elif party in ["Lean Republican", "Moderate Republican", "Strong Republican"]:
        ideology_probs = [0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.3, 0.01]
    else:
        ideology_probs = [0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.04, 0.01]

    ideology = np.random.choice(ideologies, p=ideology_probs)

    # Add some non-response for demographics
    # Replace some values with "Prefer not to say" or missing data
    non_response_rate = 0.05  # 5% chance of non-response for each field

    demographics = {
        'age': age,
        'gender': gender,
        'race': race,
        'education': education,
        'sexual_orientation': sexual_orientation,
        'party': party,
        'ideology': ideology,
        'income': income,
        'religion': religion,
        'marital_status': marital_status,
        'residential_area': residential_area
    }

    # Apply non-response to some fields
    for key in demographics:
        if random.random() < non_response_rate:
            if key == 'age':
                demographics[key] = np.nan  # Missing value for age
            else:
                demographics[key] = "Prefer not to say"

    # Assign to a message group (treatment or control)
    message_id = np.random.choice([m["message_id"] for m in messages])
    message_content = next(m["message_content"] for m in messages if m["message_id"] == message_id)

    # simulating base probability of voting for Harris based on demographics
    # First starts with party affiliation as strongest predictor
    base_prob = party_vote_probs.get(party, 0.5)

    # Adjust with other demographics
    base_prob = 0.7 * base_prob + 0.3 * ideology_vote_probs.get(ideology, 0.5)
    base_prob = 0.9 * base_prob + 0.1 * race_vote_probs.get(race, 0.5)

    age_prob = 0.5  # Default
    for (lower, upper), prob in age_brackets.items():
        if lower <= age <= upper:
            age_prob = prob
            break
    base_prob = 0.9 * base_prob + 0.1 * age_prob
    base_prob = 0.9 * base_prob + 0.1 * education_vote_probs.get(education, 0.5)

    # Apply treatment effect based on the message
    treatment_effect = message_effects[message_id]

    # treatment effects for specific demographics from the report
    if 18 <= age <= 34:
        treatment_effect += demo_message_effects["age_under_35"] * (message_id != 0)

    if race == "Black":
        treatment_effect += demo_message_effects["race_Black"] * (message_id != 0)

    if race == "Hispanic":
        treatment_effect += demo_message_effects["race_Hispanic"] * (message_id != 0)

    if ideology == "Moderate":
        treatment_effect += demo_message_effects["ideology_Moderate"] * (message_id != 0)

    # Simulate low turnout status (in reality would be based on voting history probably !!!???)
    turnout_score = np.random.randint(0, 100)
    low_turnout = turnout_score < 20

    if low_turnout:
        treatment_effect += demo_message_effects["low_turnout"] * (message_id != 0)

    # Apply treatment effect to base probability
    treatment_effect_weight = 0.5
    final_prob = min(max(base_prob * (1 - treatment_effect_weight) + treatment_effect * treatment_effect_weight, 0), 1)

    # Final vote choice
    if random.random() < 0.05:
        vote_choice = "Undecided"
    else:
        vote_choice = "Harris" if random.random() < final_prob else "Trump"

    # Create response date within the survey period
    start_date = datetime(2024, 10, 22)
    end_date = datetime(2024, 10, 25)
    response_timestamp = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )

    # Add some missing data for other fields but never for vote_choice
    data.append({
        'person_id': person_id,
        'message_id': message_id,
        'message_content': message_content,
        'age': demographics['age'],
        'gender': demographics['gender'],
        'race': demographics['race'],
        'education': demographics['education'],
        'sexual_orientation': demographics['sexual_orientation'],
        'party': demographics['party'],
        'ideology': demographics['ideology'],
        'income': demographics['income'],
        'religion': demographics['religion'],
        'marital_status': demographics['marital_status'],
        'residential_area': demographics['residential_area'],
        'turnout_score': turnout_score,
        'response_timestamp': response_timestamp,
        'vote_choice': vote_choice
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Some simple verification
print(f"Total respondents: {len(df)}")
print(df['vote_choice'].value_counts(normalize=True))

# Check treatment effects
control_group = df[df['message_id'] == 0]
control_harris_rate = (control_group['vote_choice'] == 'Harris').mean()

for mid in range(1, 6):
    treatment_group = df[df['message_id'] == mid]
    treatment_harris_rate = (treatment_group['vote_choice'] == 'Harris').mean()
    print(f"Message {mid} effect: {(treatment_harris_rate - control_harris_rate):.4f}")

# Export to CSV
# create the directory if it doesn't exist
import os
if not os.path.exists("data"):
    os.makedirs("data")
df.to_csv("data/survey_data.csv", index=False)
