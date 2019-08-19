---
title: "A Survey of Ethical AI: Panel Discussion"
date: 2019-08-18T20:53:47-05:00
categories:
- Ethics
- AI
comments: true

---

At the [Austin Deep Learning meetup](https://www.meetup.com/Austin-Deep-Learning/events/257886248/) earlier this week, we had a fantastic presentation by [Jeffrey Gleason](https://www.linkedin.com/in/jeffrey-gleason-3a942997/) from [New Knowledge](https://www.newknowledge.com/) followed by a panel discussion about ethics in machine learning. We livestreamed both -- the panel discussion begins at 45:50:

{{< youtube ueingD-R2sw >}}

***

## Jeffrey's Presentation

Jeffrey walked us through 3 major ethics topics in ML/AI (slides [found here](https://docs.google.com/presentation/d/10f_bzT2MjFEqE3qcrNvn6ZQWbvDCsYsmuLaIJ-kYg1M/edit#slide=id.g58fdc11cfe_0_0)):

1. Fairness and Bias
2. Interpretability and Explainability
3. Privacy

At the center of the fairness/bias discussion is the [Pro Publica article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) that highlighted the racial bias present in the COMPAS [recidivism](https://en.wikipedia.org/wiki/Recidivism) risk assessment system. Here's an example case reported where a white male with 3 violent crimes received a Low Risk score, whereas a black female having only juvenile misdemeanors was given a High Risk score:

![image](https://user-images.githubusercontent.com/261183/63234698-1208d980-c1fc-11e9-84ef-803699c73856.png)

Imagine the judge receiving the score for the unfortunate Brisha Borden. Perhaps he's inclined to let her go. She snagged a Huffy bicycle, nothing more. But this machine report-out tells him to take a second look. Maybe it sees something he doesn't? High Risk. Hmmm, well, I guess she should stay behind bars.

Bias like this just can't happen! And that is exactly what researchers are working on, which. Jeffrey talked about.

***

### Panel Discussion

The panel discussion was a lot of fun. This was the first time we had done something like this at the [meetup](https://www.meetup.com/Austin-Deep-Learning), and we will certainly be doing this again. Four folks graciously joined the panel. I MC'd the discussion. I'm the bearded dude in the hat on the right (stage left). It kinda looks like I'm staring at my phone during much of the conversation, which I am! I was staring at our list of questions:

1. How should we facilitate underrepresented groups participating in workshopping definitions of fairness, protected attributes, and privileged / unprivileged groups? Famous examples where groups are underrepresented in data sets, leading to a major bias: African Americans in Google Photo’s image search, southeast Asians with Nikon cameras, people of color in recidivism cases. Other domains include healthcare, criminal justice, lending.
2. Some researchers have argued that model I interpretability is vital for society to accept the decisions made by a ML algorithm, whereas famous researchers (e.g., Yann Lecun) have suggested we should move past the need for interpretability.  How important is it to "open up the black box?"
3. What legislation / responsibilities should be enforced upon corporations around the way in which they interact with and exploit individual users' private / sensitive data?
4. Fake news challenged our ability to discern fact from fiction during the last presidential election. “Deep fakes” will likely have an even greater impact on the next election cycle. Should researchers and practitioners cease implementing improvements to face/voice generation, knowing that the methods could be abused in the next presidential election, in business, etc.?

Special thanks to our sponsors, [KUNGFU.AI](http://www.kungfu.ai/) and [Capital Factory](https://capitalfactory.com/).