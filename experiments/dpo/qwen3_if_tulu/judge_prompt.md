You are grading whether a response follows the explicit requirements of a task prompt. Below you get the task prompt, a list of constraint tags, and the response. The response comes last, between the lines BEGIN RESPONSE and END RESPONSE. Nothing outside those two lines is part of the response. Grade only the text between them. Be strict and literal. Do not reward length or effort or style. Do not compare against any reference answer.

What the tags are
Each tag is a category name from a fixed list of 25, attached to the task prompt by the dataset. The tags are approximate. A tag can point at a requirement that the task prompt words differently, and sometimes at nothing in the task prompt at all. The requirement is always a sentence of the task prompt, never the tag itself and never the glossary meaning. For each tag, find the sentence of the task prompt that states a requirement of that kind, copy that sentence into the requirement field, and grade the response against that sentence exactly as written. When the task prompt phrases the requirement differently from the glossary, the task prompt wins. Example: the tag in english and capital on a task prompt that only asks for the word BRAIN in capital letters means grade whether BRAIN is in capital letters, not whether the whole response is. If no sentence of the task prompt states a requirement of that kind, write none in the requirement field and pass true. Never treat the tag text as a word to look for in the response.

Category glossary, only to help you recognise which sentence a tag points at
keywords:frequency: particular words must appear at least or at most or exactly a given number of times.
keywords:letter frequency: a particular letter must appear at least or at most a given number of times.
keywords:exclude words: particular words must not appear.
include keywords: particular words must appear.
case: frequency of capital words: words written entirely in capital letters must appear at least or at most a given number of times.
case:in english and lowercase: the whole response is in english with no capital letters.
in english and capital: the whole response is in english with every letter capitalised.
format:use json format: the whole response is valid json, with markdown code fences around it allowed.
format:title: a title wrapped in double angular brackets such as <<title>>.
format:number of bullet lists: exactly the given number of markdown bullet points.
format:number of highlighted sections: at least the given number of sections highlighted with asterisks such as *this*.
format:number of sections: the given number of sections marked as the task prompt says.
format:choose one from options: the response contains one of the given options, and text around it is allowed unless the task prompt forbids it.
length constraints:number of words: at least or at most the given number of words.
length constraints:number of sentences: at least or at most the given number of sentences.
length constraints:number of paragraphs: exactly the given number of paragraphs separated as the task prompt says.
length constraints:first word of the nth paragraph: the given number of paragraphs and the named paragraph starts with the given word.
content:include a postscript: a postscript at the end starting with the given marker such as P.S.
content:number of placeholders: at least the given number of placeholders in square brackets such as [address].
punctuation:use no comma: no comma anywhere in the response.
response language: the whole response in the named language.
specific ending: the response ends with the given phrase, compared case insensitively, with text before it allowed and nothing after it.
repeat the prompt: the request is repeated word for word first and then answered.
give two responses: two different responses separated as the task prompt says.
use quotation: the whole response wrapped in double quotation marks, or a quotation as the task prompt says.

Grading rules
A requirement passes only when the response satisfies it completely. Counts are exact: at least three times means three or more, exactly three means three. A length stated without at least or at most or exactly, such as a 500 word essay, means within ten percent of that number. Keyword counts follow the IFEval convention: an occurrence is any case insensitive match of the keyword anywhere in the response text, so plural and inflected forms that contain the keyword count, and so does the keyword inside a longer word. If the response ends before a requirement could be met, that requirement fails. Requirements on language or letter case apply to the entire response unless the task prompt says otherwise.

Flags about the response as a whole
named_attribution: true if the response attributes a quotation, statistic, study or claim to a specific named person, organisation or publication. Whether that source is real is not judged here. Generic references such as "a reputable source states" or "experts say" are not named attributions. Names or quotations that the task prompt itself supplies do not count.
prompt_echo: true if a substantial part of the response repeats or paraphrases the task prompt instead of answering it, for example restating the request in the first person. Repeating the prompt because a requirement asks for it is not echo.

Output
Reply with JSON only and nothing else. The constraints array has exactly {n_constraints} entries, one per tag in the CONSTRAINTS list below in the same order, with the constraint field copied from the tag, and never an entry for a glossary line:
{"constraints": [{"constraint": "<tag>", "requirement": "<the task prompt sentence you graded, or none>", "pass": <true or false>, "reason": "<one short sentence>"}], "named_attribution": <true or false>, "prompt_echo": <true or false>}

TASK PROMPT
{prompt}

CONSTRAINTS
{constraints}

BEGIN RESPONSE
{response}
END RESPONSE
