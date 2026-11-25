What the methods get:

The cleaned, redacted activity for a time window. Each segment has:
a segment ID and start/end times
the app (e.g. VS Code) or website (e.g. cs61a.org)
the cleaned-up page or window title
the cleaned-up URL
when you were away, either shaded in or already removed

The label list: label names, what each one covers, and typical apps. Only the LLM reads these directly. The sequence-mining method uses its own rules for mapping apps and websites to labels, built from the same list.
Their own settings: thresholds, the prompt, and the model name. None of these come from your log.
For finding repeats: the method’s own earlier results, such as its summaries from previous days. Never your “Repeat of” column.


What they never get: 

anything from the task log:
start and end times, and therefore how many tasks are in a window
labels, steps or the apps/sites you listed
“Repeat of”, “Interrupted?” or Notes


Which time window to use
The default should be a whole session or a whole day. A session means continuous activity between long breaks, like sleep or 20+ minutes away. Your logged tasks fall somewhere inside, mixed with unlogged time, and the methods have to work out the boundaries themselves. Their goal is to reconstruct the workflow.

Padding a task window by ±20 minutes, which the brief offers as an option, still gives some of the answer away. The method knows a task sits roughly in the middle, and it probably assumes there’s only one task in the window. Use it only as a quick debugging view.
When I benchmark next week, each method runs over whole days or sessions. You then score only its output that overlaps your logged tasks, and ignore whatever it says about unlogged time.


RULES:
Don’t feed the task log to the methods. It also mustn’t cut the input at your exact logged start and end times, since that would hand the methods the answer. The label list is allowed as input.
Treat T01–T17 as the development set. Tune only on these; tasks you log later are the test set.
Never commit raw data or send unredacted text to an LLM.