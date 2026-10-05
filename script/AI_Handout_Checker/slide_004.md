So how does it actually work: you type a single command, python nano dot py, point it at the handout P D F, and a three stage pipeline does the rest.
First, stage A reads the page, where a vision language O C R model turns the scan into text, handwriting included, and a computer vision detector catches the hand drawn circles the model would miss.
Then stage B grades it, with a Qwen three model splitting the page into exercises and judging each answer, while plain Python recomputes every score so the model never owns the final mark.
Finally, stage C writes the results back on the page, anchoring each one to its exact row with Tesseract and drawing red ticks, crosses and corrections beside the answers.
What comes out is a graded P D F marked in red, plus a class C S V with weak spot statistics that show where the whole class is struggling.
And because the two A I models are loaded one at a time, it all fits in eight gigabytes of G P U memory, so let's take a closer look at that first reading stage.
