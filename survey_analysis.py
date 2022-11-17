# Initial imports and environment setting.

%matplotlib inline
import numpy as np
import pandas as pd
from textwrap import wrap

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import seaborn as sns
sns.set(context='talk', style='white')

import warnings
warnings.filterwarnings('ignore')


def survey_monkey_data_long(df: pd.DataFrame, id_col: str = 'Unnamed: 0'):
    """Uses pd.DataFrame of SurveyMonkey data and transforms it into long-format.

    Args:
        df (pd.DataFrame): Raw SurveyMonkey data.
        id_col (str): Column that has the participant Id / number, in case its renamed.
    Returns:
        pd.DataFrame: SurveyMonkey data in long format.
    """
    survey_clean = df.copy()

    # Useful columns as ID:
    ID_cols = [id_col,
               'Are you a member of OHBM?',
               'What geographic region are you currently located in?',
               'What is your current career status?']

    # Define questions with multiple answer options
    multi_Q1 = ('Which of the following platforms do you use to access OHBM content? '
                + 'When applicable, a direct link to the platform is provided next to each option. '
                + 'Please check all options that apply.')

    multi_Q2 = ('Do you currently follow any of the following OHBM Special Interest Groups '
                + '(SIG) platforms? When applicable, a direct link to the platform is provided '
                + 'next to each option. Please check all options that apply.')

    multi_Q3 = 'How important is each of these types of content to you?'

    # Add questions to list and define the range of "Unnamed: XX" answer options.
    multi_Qs = [multi_Q1, multi_Q2, multi_Q3]
    unnamed_range = [(13, 21), (41, 47), (49, 55)]

    orig_name = []
    new_name = []

    for q, ur in zip(multi_Qs, unnamed_range):
        unnamed = [q] + [f'Unnamed: {i}' for i in range(ur[0], ur[1] + 1)]
        renamed = [q + '@@' + df.loc[0, un] for un in unnamed] # Use @@ as separator
        orig_name.extend(unnamed)
        new_name.extend(renamed)

    # Rename "Unnamed: XX" Columns to have the name of the Question.
    rename_dict = {i : j for i, j in zip(orig_name, new_name)}
    survey_clean.rename(columns=rename_dict, inplace=True)
    # Drop response description (stored in first row)
    survey_clean.drop(0, inplace=True)

    # Bring into long format:
    survey_long = survey_clean.melt(id_vars=ID_cols, var_name='questions',
                                    value_name='response')
    survey_long.sort_values(id_col, inplace=True)

    survey_long['questions'] = survey_long['questions'].str.split('@@').str[0]

    return survey_long


def survey_data_wide(df_long: pd.DataFrame, id_col: str = 'Unnamed: 0'):
    """Moves survey data from long format back to wide format. Summarizes multi
    answer questions into lists.

    Args:
        df_long (pd.DataFrame): SurveyMonkey data in long format.
        id_col (str): Column that has the participant Id / number, in case its renamed.

    Returns:
        pd.DataFrame: SurveyMonkey data in wide format.
    """

    ID_cols = [id_col,
               'Are you a member of OHBM?',
               'What geographic region are you currently located in?',
               'What is your current career status?']

    survey_wide = df_long.copy()
    survey_wide[id_col] = survey_wide[id_col].astype("string")
    # Drop nan responses
    survey_wide.dropna(inplace=True, subset=['response'])
    # Summarize answers to questions in list
    survey_wide = survey_wide.groupby(ID_cols + ['questions']).agg({'response': lambda x: list(x)}).reset_index()
    # Move ID cols into a single column - to avoid duplicates for pivoting
    survey_wide['new_index'] = survey_wide[ID_cols].apply(lambda row: '@@'.join(row.values.astype(str)), axis=1)
    survey_wide.drop(columns=ID_cols, inplace=True)
    # Pivot
    survey_wide = survey_wide.pivot('new_index', columns=['questions'], values='response').reset_index()
    # Move temporary index back into columns
    for n, id in enumerate(ID_cols):
        survey_wide[id] = survey_wide['new_index'].str.split('@@').str[n]

    survey_wide[id_col] = survey_wide[id_col].astype("int")

    survey_wide.drop(columns='new_index', inplace=True)
    survey_wide.sort_values(id_col, inplace=True)

    return survey_wide


# Load in the data. This spreadsheet is (almost) direct from SurveyMonkey,
# though IP addresses, access dates, and free-text responses were scrubbed
# to anonymize respondents.

df = pd.read_csv('public_survey_data.csv', sep=',')

# SurveyMonkey provides an odd nesting of responses when exporting results.
# We'd like to convert this structure to a pandas MultiIndex data frame.
# First, let's find question indices -- adapted from https://stackoverflow.com/a/49584888

indices = [i for i, c in enumerate(df.columns) if not c.startswith('Unnamed')]
slices = [slice(i, j) for i, j in zip(indices, indices[1:] + [None])]
repeats = [len(range(*slice.indices(len(df.columns)))) for slice in slices]

# Now let's grab all of the questions and each of the options provided as possible responses.

questions = [c for c in df.columns if not c.startswith('Unnamed')]
options = df.iloc[:1].values[0].tolist()

# We can pair each possible response with its associated question...

matched_questions = []
for index, question in enumerate(questions):
    matched_questions += [question] * repeats[index]

# ...and create a new dataframe named 'responses' that correctly pairs questions and responses.

index = pd.MultiIndex.from_arrays([matched_questions, options],
                                  names=('question', 'options'))
data = df.iloc[2:].values
responses = pd.DataFrame(data=data, columns=index)

# First demographic questions -- the normalize keyword converts to percentages.
responses['Are you a member of OHBM?',
          'Response'].value_counts(normalize=True)

responses['What geographic region are you currently located in?',
          'Response'].value_counts(normalize=True)

responses['What is your current career status?',
          'Response'].value_counts(normalize=True)


def plot_stacked_bar(df, figwidth=12.5, textwrap=30):
    """
    A wrapper function to create a stacked bar plot.
    Seaborn does not implement this directly, so
    we'll use seaborn styling in matplotlib.

    Inputs
    ------

    figwidth: float
        The desired width of the figure. Also controls
        spacing between bars.

    textwrap: int
        The number of characters (including spaces) allowed
        on a line before wrapping to a newline.
    """
    reshape = pd.melt(df, var_name='option', value_name='rating')
    stack = reshape.rename_axis('count').reset_index().groupby(['option', 'rating']).count().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(15, figwidth))
    bottom = np.zeros(len(stack['option'].unique()))
    clrs = sns.color_palette('Set1', n_colors=4)  # to do: check colorblind friendly-ness
    # labels = ['Not aware', 'Aware but not engage',
    #           'Aware and occassionally engage', 'Aware and engaged']
    labels = ['Unsure', 'Not relevant', 'Somewhat relevant',
              'Relevant only to annual meeting', 'Consistently relevant']

    for i, rating in enumerate(np.unique(stack['rating'])):
        stackd = stack.query(f"rating == '{rating}'")
        ax.barh(y=stackd['option'], width=stackd['count'], left=bottom,
                tick_label=['\n'.join(wrap(s, textwrap)) for s in stackd['option']],
                color=clrs[i], label=labels[i])
        bottom += stackd['count'].to_numpy()

    sns.despine()
    ax.set_xlabel('Count', labelpad=20)
    ax.legend(title='Rating', bbox_to_anchor=(1, 1))

    return ax


# Next, we'd like to look at results
# 'How would you describe the content on the OHBM job board?',

content_questions = [
    'How would you describe the content in OHBM emails?',
    'How would you describe the content in the OHBM blog?',
    'How would you describe the content on OHBM Twitter?',
    'How would you describe the content in the NeuroSalience podcast?',
    'How would you describe the content in OHBM Facebook?',
    'How would you describe the content in OHBM YouTube?',
    'How would you describe the content in OHBM LinkedIn?',
    'How would you describe the content in OHBM OnDemand?',
    ]

content_responses = responses[content_questions]
ax = plot_stacked_bar(content_responses, figwidth=15, textwrap=35)
fig = ax.figure
fig.savefig('content.png', dpi=150, bbox_inches='tight')
