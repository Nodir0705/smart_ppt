Behind every handout marked in red sits one command and a three stage pipeline.
First, stage A reads printed text and handwriting, while a detector catches circled answers.
Then stage B judges each answer, and plain Python recomputes every score, not the model.
Finally, stage C draws red ticks, crosses and corrections on the page, giving a graded PDF and class CSV.
The models load one at a time, so everything fits in eight gigabytes of GPU memory.
