You are grading whether a response follows the explicit constraints of a task. Be strict and literal. Judge only the response text shown below. Do not reward length or effort or style. Do not compare against any reference answer.

TASK PROMPT
{prompt}

CONSTRAINTS
{constraints}

RESPONSE
{response}

How to grade
Each constraint label above names one requirement that is spelled out in the task prompt. Find that requirement in the task prompt and check the response against it exactly as written there. A constraint passes only when the response satisfies it completely. Counts are exact: at least three times means three or more, exactly three means three. If the response ends before a requirement could be met, that requirement fails. Constraints on language or letter case apply to the entire response unless the task prompt says otherwise.

Then set two flags about the response as a whole.
fabricated_attribution: true if the response attributes a quotation, statistic, study or claim to a specific named person, organisation or publication and presents it as real. Generic references such as "a reputable source states" or "experts say" are not fabrication. Names or quotations that the task prompt itself supplies are not fabrication.
prompt_echo: true if a substantial part of the response repeats or paraphrases the task prompt instead of answering it, for example restating the request in the first person. Repeating the prompt because a constraint asks for it is not echo.

Reply with JSON only and nothing else, in exactly this form, with one entry per constraint in the same order as the list above and the constraint field copied from the label:
{"constraints": [{"constraint": "<label>", "pass": <true or false>, "reason": "<one short sentence>"}], "fabricated_attribution": <true or false>, "prompt_echo": <true or false>}
