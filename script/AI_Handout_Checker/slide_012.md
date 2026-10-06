For the curious, this appendix looks under the hood of that private, local grading.
Nanonets reads the pages and a small Qwen model grades them, taking turns on an eight gigabyte card.
Classic computer vision spots the circled answers, with zero false positives in testing.
Then Python recomputes every score, so the same worksheet always earns the same grade.
Finally, disk caching and a fully offline stack keep it fast and private, which wraps up our tour.
