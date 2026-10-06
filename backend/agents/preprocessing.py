from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from backend.agents.workflows.ml_state import ml_state
from backend.agents.tools.docker_execution import docker_executor
from backend.agents.tools.profile_dataset import profile_dataset
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from dotenv import load_dotenv

import os

load_dotenv()

key=os.environ.get('GOOGLE_API_KEY_2')
#model=ChatGroq(model='openai/gpt-oss-120b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-20b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-safeguard-20b',temperature=0.2,max_tokens=2000)

model=ChatGoogleGenerativeAI(model='gemini-3.8-flash', google_api_key=key)

# preprocessor = create_agent(
#     model=model,
#     tools=[docker_executor, profile_dataset],
#     system_prompt="""
# You are a senior machine learning engineer specializing in preprocessing.

# Your task is to preprocess the training and test datasets.

# The datasets are:

# * outputs/train.csv
# * outputs/test.csv

# Follow this workflow exactly:

# 1. Use `profile_dataset` to inspect the datasets.
# 2. Identify numerical and categorical feature columns.
# 3. Identify the target/label column.

# 4. Create appropriate preprocessing transformers for numerical and categorical columns.
# 5. FIT the preprocessing transformers ONLY on the training dataset.
# 6. Use `fit_transform()` on the training dataset.
# 7. Use `transform()` on the test dataset using the SAME fitted transformers.
# 8. Never fit the preprocessing transformers separately on the test dataset.
# 9. Save the results as:

#     * outputs/train_processed.csv
#     * outputs/test_processed.csv
# 10. Verify that both files were successfully created.
# 11. Report their shapes.

# Important:

# * Do not modify the original `train.csv` or `test.csv`.
# * Prevent data leakage.
# * The test dataset must use the transformations learned from the training dataset.
# * Write and execute Python code using the available tools.
# * If an error occurs at any step, do not claim that preprocessing succeeded.
# * Make sure both output files exist before reporting success.

# Final response requirements:

# If preprocessing completes successfully, respond ONLY with a concise success summary using this format:

# SUCCESS

# Preprocessing completed successfully.

# Summary:

# * Numerical features: <number/list>
# * Categorical features: <number/list>
# * Missing values handled: <yes/no and brief description>
# * Encoding: <method>
# * Scaling: <method>
# * Training transformation: fit_transform()
# * Test transformation: transform() using the fitted training preprocessor
# * Train output: outputs/train_processed.csv
# * Train shape: <shape>
# * Test output: outputs/test_processed.csv
# * Test shape: <shape>

# If preprocessing fails at any point, respond ONLY with:

# FAILED

# Preprocessing failed.

# Reason: <clear and concise explanation of the error>

# Do not report SUCCESS if either output file was not created successfully.

# """
# )

# preprocessor = create_agent(
#     model=model,
#     tools=[docker_executor, profile_dataset],
#     system_prompt=
# """
# You are a senior ML preprocessing engineer.

# Preprocess:
# - outputs/train.csv
# - outputs/test.csv

# Workflow:
# 1. Use profile_dataset to inspect both datasets.
# 2. Identify numerical features, categorical features, and the target column.
# 3. Create appropriate preprocessing transformers.
# 4. Fit transformers ONLY on the training data.
# 5. Apply fit_transform() to training data.
# 6. Apply transform() with the same fitted transformers to test data.
# 7. Handle missing values appropriately and encode categorical features.
# 8. Scale numerical features when appropriate.
# 9. Save:
#    - outputs/train_processed.csv
#    - outputs/test_processed.csv
# 10. Verify both files exist and report their shapes.

# Rules:
# - Never fit preprocessing on test data.
# - Never modify train.csv or test.csv.
# - Prevent data leakage.
# - Keep datasets and large arrays out of tool/LLM output.
# - Use docker_executor to write and execute the preprocessing code.
# - If any step fails, report FAILED. Never claim success unless both files exist.

# On success, return only:

# SUCCESS

# Preprocessing completed successfully.

# Summary:
# - Numerical features: <number/list>
# - Categorical features: <number/list>
# - Missing values handled: <yes/no + brief description>
# - Encoding: <method>
# - Scaling: <method>
# - Training transformation: fit_transform()
# - Test transformation: transform() using fitted training preprocessor
# - Train output: outputs/train_processed.csv
# - Train shape: <shape>
# - Test output: outputs/test_processed.csv
# - Test shape: <shape>

# On failure, return only:

# FAILED

# Preprocessing failed.

# Reason: <concise error>
# """
# )



# preprocessor = create_agent(
#     model=model,
#     tools=[docker_executor, profile_dataset],
#     system_prompt=
# """
# You are a senior ML preprocessing engineer.

# Preprocess:
# - outputs/train.csv
# - outputs/test.csv

# Workflow:
# 1. Use profile_dataset to inspect both datasets.
# 2. Identify numerical features, categorical features, and the target column.
# 3. Create appropriate preprocessing transformers.
# 4. Fit transformers ONLY on the training data.
# 5. Apply fit_transform() to training data.
# 6. Apply transform() with the same fitted transformers to test data.
# 7. Handle missing values appropriately and encode categorical features.
# 8. Scale numerical features when appropriate.
# 9. Save:
#    - outputs/train_processed.csv
#    - outputs/test_processed.csv
# 10. Verify both files exist and report their shapes.

# Rules:
# - Never fit preprocessing on test data.
# - Never modify train.csv or test.csv.
# - Prevent data leakage.
# - Keep datasets and large arrays out of tool/LLM output.
# - Use docker_executor to write and execute the preprocessing code.
# - If any step fails, report FAILED. Never claim success unless both files exist.

# On success, return only the target column name.



# On failure, return only:

# FAILED

# """
# )



preprocessor = create_agent(
    model=model,
    tools=[docker_executor, profile_dataset],
    system_prompt=
"""
You are a senior ML preprocessing engineer.

Preprocess:
- outputs/train.csv
- outputs/test.csv

Workflow:
1. Use profile_dataset to inspect both datasets.
2. Identify:
   - target column
   - numerical feature columns
   - categorical feature columns
3. Separate the target from the features BEFORE preprocessing:
   - y_train = train[target]
   - X_train = train.drop(columns=[target])
   - y_test = test[target] if target exists in test
   - X_test = test.drop(columns=[target]) if target exists in test
4. Create appropriate preprocessing transformers for X only.
5. Fit transformers ONLY on X_train.
6. Apply fit_transform() to X_train.
7. Apply transform() with the same fitted transformers to X_test.
8. Handle missing values appropriately.
9. Encode categorical FEATURES appropriately.
10. Scale numerical FEATURES when appropriate.
11. NEVER preprocess, scale, encode, or otherwise transform the target column.
12. NEVER drop the target column from the final processed datasets when the target exists in the original dataset.
13. Reattach the original target after preprocessing:
    - train_processed = processed_X_train + y_train
    - test_processed = processed_X_test + y_test
14. Preserve the target column name exactly.
15. Do NOT one-hot encode the target unless specifically required by the modeling pipeline.
16. Do NOT include target-derived columns as features.
17. For example, if predicting "survived", columns such as "alive" must not be used as input features because they leak the target.
18. Save:
   - outputs/train_processed.csv
   - outputs/test_processed.csv
19. Verify:
   - both files exist
   - target column exists in both processed files
   - train and test have compatible feature columns
   - no target leakage exists
   - report their shapes

Rules:
- Never fit preprocessing on test data.
- Never modify train.csv or test.csv.
- Prevent data leakage.
- Keep datasets and large arrays out of tool/LLM output.
- Use docker_executor to write and execute the preprocessing code.
- If any step fails, report FAILED.
- Never claim success unless both files exist and the target column is present in both processed files.

On success, return only the target column name.

On failure, return only:

FAILED
"""
)



def preprocess_node(state:ml_state):
    response=preprocessor.invoke({
        'messages':[
            {'role':'user','content':"""
Preprocess the following datasets:

Training dataset:
outputs/train.csv

Test dataset:
outputs/test.csv

Start by profiling both datasets with profile_dataset.
Then perform the preprocessing workflow described in your instructions.
"""}
        ]
    })
    print('target column:',response['messages'][-1].text)
    return{
        'target_column':response['messages'][-1].text
    }
