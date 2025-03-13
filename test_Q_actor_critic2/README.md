Here is an Q-function actor-critic algorithm with a buffer made to reach a program that produces a desired interference. 
The desired interference is a 3-dimensional point, with component of very different scales. Therefore, we introduce a weighted distance is the behavior space.
During the training, each episode start with a random program, with a random number of lines from 1 to 5. During the training, we want the episodes to be shorter and shorter.

The reward is chosen as the negativ weighted distance between the state (program) and the objective. 

Below, we represent the number of step per episodes during the training phase

![Alt text](image/nb_iterations.png)
