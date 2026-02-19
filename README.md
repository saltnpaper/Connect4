# Connect4
[You can play against the connect4 here] (https://anvil.works/build#clone:HD2M4NJEKKLE5FUS=ZM4EQ6UHT7KAB27UMEKVCUPI)

### Overview
This project presents a comparative analysis of two distinct neural network architectures—a Convolutional Neural Network (CNN) and a Transformer—trained to master the game of Connect 4.

Rather than relying on basic heuristics, we developed a robust "teacher" dataset of 33,000 games using Monte Carlo Tree Search (MCTS) self-play. By deploying these models into a live "Battle Arena," we explored the trade-offs between the CNN's localized spatial intuition and the Transformer's global sequence-mapping capabilities.

#### The Full-Stack Architecture
The project is a complete end-to-end system:

* **Data Pipeline:** MCTS game generation with horizontal flipping for data augmentation.

* **Modeling:** Custom PyTorch implementations of a "Wide & Deep" CNN and a Sinusoidal Positional Encoding Transformer.

* **Deployment:** The models are hosted on a Dockerized AWS backend.

* **Interface:** A real-time web frontend built with Anvil, allowing users to play against the bots directly in their browser.
