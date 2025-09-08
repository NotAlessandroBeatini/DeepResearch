Think of it this way:

The Researcher is a collector. It gathers raw materials. 📚

The Synthesizer is a thinker. It takes those raw materials and creates something meaningful out of them. 🤔

It's the component that performs the "deep" part of your "deep research agent."

## How it Fits into Your Workflow
Your workflow becomes a multi-stage assembly line for knowledge:

Planner ➡️ Researcher ➡️ Synthesizer

Planner: Sets the high-level goal.

Output: A research question, like "What are the latest techniques in large language model quantization?"

Researcher (The Collector): Executes the plan by finding and gathering raw information. It uses the atomic tools (add_scholar_reference, etc.) to populate the blackboard.

Output: A list of Citation and Reference objects on the blackboard. Crucially, these objects only have the basic metadata filled out (title, URL, authors). The fields requiring deep analysis (extended_summary, contributions) are still empty (null).

Synthesizer (The Thinker): This agent (or a separate process) activates after the Researcher is done. It takes the Researcher's output as its input and performs the high-value cognitive work.

Its Job:

Enrich the Data: It goes through each Citation on the blackboard, reads the actual source document (via its URL or a retrieval tool), and fills in the previously empty fields: extended_summary, contributions, results.

Find Connections: It looks at all the collected sources as a whole. It identifies common themes, conflicting findings, and the overall progression of ideas in the field.

Generate Insights: This is where your Finding model finally comes into play! The Synthesizer's ultimate goal is to create Finding objects (e.g., statement: "Quantization-Aware Training consistently outperforms Post-Training Quantization for models under 7B parameters.") and back them up with the sources the Researcher found.

## Why Bother With a Separate Agent?
This separation of concerns is what makes the system powerful and reliable.

Focus: The Researcher agent has one simple job: find relevant links and metadata. It's fast and less likely to get confused.

Context: The Synthesizer has the benefit of seeing all the collected research at once. This allows it to make much better summaries and identify deeper connections than an agent trying to do everything on the fly.

Capability: You can use a more powerful (and maybe more expensive) LLM for the Synthesizer, since it performs a more complex task. The Researcher can be a cheaper, faster model.
