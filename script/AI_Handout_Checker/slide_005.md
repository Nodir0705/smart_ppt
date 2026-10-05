The first of those three stages is stage A, reading, and the hard part here is a messy scan with printed text and pen handwriting on the same page.
First, a vision language O C R model, Nanonets, reads the whole page in one pass and turns it into structured text, tables, checkboxes and handwritten fill ins together.
You can see that on the example scan, where the student's handwritten verb forms come through right inside the printed tenses chart.
Next, we tuned the scan resolution for real handouts, landing on the sweet spot where faint pencil corrections survive without garbling the printed text.
Finally, circled answers need real vision, because the language model cannot reliably tell when nothing is circled, so a computer vision detector finds the drawn loops and the model just names the word inside.
With the page read, the next stage can turn those answers into a grade.
