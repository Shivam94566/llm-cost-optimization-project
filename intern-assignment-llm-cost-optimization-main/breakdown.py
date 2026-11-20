from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

for name, cls, tid in [("Q1", SupportCopilot, "T-1001"),
                       ("Q2 physical", SupportPipeline, "T-1004"),
                       ("Q2 escalation", SupportPipeline, "T-1003")]:
    a = cls(); a.handle_ticket(tid)
    for c in a.session.calls:
        print(name, tid, c.label, "input:", c.input_tokens, "output:", c.output_tokens, "points:", c.points())