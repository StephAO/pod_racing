## POD RACING ##
Genetic algorithm for [pod racing game](https://www.codingame.com/multiplayer/bot-programming/coders-strike-back), ranking 242nd in the legend league.  
Followed [Magus' blog](http://files.magusgeek.com/csb/csb_en.html) for the basis of my algorithm.  
Developed in python 2.7 on Ubuntu.

#### Post Mortem ####
- Eugenics
  - Elitism (keep best pod in every iteration, so that there is no backward progress).
  - Breed smartly (bred the individual with the best runner score with the individual with the best bruiser score)   
- Don't change too much in a single mutation
  - The more you change at once, the less likely you are to find improvement (but the potential for improvement is larger).
  - Making smaller mutations creates more consistent (but more incremental) improvement, especially important when there isn't enough time to run long simulations.
- Start with a good base population (use basic logic to create a decent starting place instead of random if possible)
  - Potential downside if base population is close to ok solutions, but far from the optimal solutions
- A good simulation is necessary
