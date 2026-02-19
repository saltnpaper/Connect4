# Connect4
[You can play against the connect4 here] (https://anvil.works/build#clone:HD2M4NJEKKLE5FUS=ZM4EQ6UHT7KAB27UMEKVCUPI)

### Overview
This project presents a comparative analysis of two distinct neural network architectures—a Convolutional Neural Network (CNN) and a Transformer -trained to master the game of Connect 4.

Rather than relying on basic heuristics, we developed a robust "teacher" dataset of 33,000 games using Monte Carlo Tree Search (MCTS) self-play. By deploying these models into a live "Battle Arena," we explored the trade-offs between the CNN's localized spatial intuition and the Transformer's global sequence-mapping capabilities.

#### The Full-Stack Architecture
The project is a complete end-to-end system:

* **Data Pipeline:** MCTS game generation with horizontal flipping for data augmentation.

* **Modeling:** Custom PyTorch implementations of a "Wide & Deep" CNN and a Sinusoidal Positional Encoding Transformer.

* **Deployment:** The models are hosted on a Dockerized AWS backend.

* **Interface:** A real-time web frontend built with Anvil, allowing users to play against the bots directly in their browser.

### Methodology: Data Generation

To train a robust neural network, we didn't want to rely on random noise; we needed a "teacher" capable of looking ahead. We generated a diverse dataset of approximately 33,000 games using Monte Carlo Tree Search (MCTS) self-play. For the training set, we produced 30,000 games where the MCTS agent played against itself with varying simulation counts ranging from 100 to 1500, averaging around 800–1000. This variance was intentional and crucial: by exposing the model to both "perfect" play and "human-like" mistakes, we prevented it from becoming brittle or only recognizing optimal lines. To keep our benchmarks honest, we generated a separate testing set of 3,000 games at a fixed difficulty of roughly 800 simulations.

To squeeze every bit of value out of this data, we leveraged the natural geometry of the board. Since Connect Four is symmetric, a winning strategy on the left is identical to one on the right. We applied horizontal flipping to every board state, effectively doubling our training data points overnight. This didn't just give us more data; it enforced "spatial invariance," teaching the model that the logic of the game holds true regardless of which side of the board the pieces are on.

### CNN Model Architecture

**Data Encoding Strategy:** The 6x7x2 Tensor While it is simpler to encode the board as a single flat 6x7 grid using +1 for the AI, -1 for the opponent, and 0 for empty spaces, we deliberately chose to encode the board as a 6x7x2 tensor. In a single-grid setup, positive and negative values can artificially cancel each other out during the convolution operation, blurring the network's ability to distinguish between its own pieces and the enemy's. By splitting the board into two distinct channels Channel 0 for the AI's pieces and Channel 1 for the Opponent's pieces, we guarantee that the convolutional filters can learn independent, clean feature maps for offensive and defensive patterns without mathematical interference.

#### The Setup
We designed a custom Convolutional Neural Network (CNN) specifically optimized for the spatial constraints of Connect Four. The input is our 6x7x2  tensor. The network consists of four convolutional blocks followed by a dense classification head:
* **Layer 1 & 2 (Feature Extraction):** These initial layers utilize 128 filters with standard 3x3 kernels to extract low-level board features. We apply Batch Normalization (BatchNorm2d) here to stabilize the gradient flow and use ReLU activation.
* Layer 3 (Pattern Detection):** We increase the depth to 256 filters and utilize a 4x4 kernel with a padding of 2.
* Layer 4 (Feature Aggregation):** A final convolutional layer with 256 filters and a 3x3 kernel aggregates the features into high-level strategic evaluations.
* Fully Connected Head:** The output is flattened and passed through a large Linear Layer of 1024 neurons, concluding with a final layer producing 7 logits that represent the probability of winning for each column drop.

#### Results and the "Battle Arena"

Training with the Adam optimizer and CrossEntropyLoss, we watched the model learn the rules of the game in real-time. We observed a steady decrease in loss, with the Validation Accuracy eventually stabilizing at approximately 63% - a strong signal given that random guessing only yields about 14%. But accuracy metrics only tell half the story, so we deployed the CNN into a live "Battle Arena" against MCTS agents. 

The results were telling: against Novice and Intermediate bots (MCTS < 800), the CNN was dominant, achieving a win rate of 65%–70%. It proved it had successfully distilled the "teacher's" knowledge. However, against Advanced agents (MCTS > 850), the win rate settled between 35%–40%. This drop-off highlights the reality of supervised learning: while the model is excellent at imitating successful patterns it has seen, it struggles to "calculate" deep, novel traps that require a look-ahead horizon deeper than its training examples.

### Transformer Model

#### Training Architecture: 
#### Setting It Up

First, we define the Sinusoidal Positional Encoding. The model uses these sine and cosine waves to understand the structure of the blocks and where each token sits in the sequence. By using the torch.zeros function, we generate a blank array that takes our maximum length and model dimensions as arguments. Then, we can arrange the position of each of these zeros using the .arange function to create a sequence of indices. Following this, we define the specific rhythm of the encoders by filling the even and odd columns of that array using torch.sin and torch.cos.
Following this, the forward function , or the flow of how these encodings actually hit the data ,needs to be defined. This is pretty straightforward: first, we scale our input embeddings by their square root to make sure the signal stays strong. Then, we simply add our positional "map" to the embeddings that our model spits out. This happens right before the data heads into the self-attention layer, giving the model the spatial context it needs to understand the order of the board.

#### The Fun Part

Now we need to specify our hyperparameters and the structure of our Transformer. Since we are using PyTorch, we first settle on the layers and then define their execution order. We start with a fully connected layer using nn.Linear() that takes our chunked board—flattened into 42 tokens—and projects them into vectors of length 128 to match what the Transformer expects. Following this, we call our PositionalEncoding class to give the model a sense of where each cell is on the grid. Our self-attention mechanism is defined with 8 heads, repeated across 3 encoder layers, with a dropout rate of 0.1 to prevent overfitting. Lastly, since a Connect Four board has 7 possible moves, our final layer is an MLP with 7 output classes. We can add Softmax for easier interpretation, or simply let the model pick the highest logit value; either way works.

In the forward pass, the data follows a specific flow. The raw board vectors first pass through the embedding layer, which spits out those 128-length vectors. Then, the sinusoidal positional encodings are added. Next, these "processed" vectors are fed into the Transformer encoder, which handles the heavy lifting of figuring out the relationships between the pieces on the board. Finally, we flatten the entire sequence and feed it into the final fully connected layer to get our answer. This approach is a bit "greedy"; by flattening all of our vectors instead of averaging them or using a class token, we retain every bit of spatial information the Transformer has processed. While we pay the price with a higher weight count in the final MLP, we believe this trade-off is worth it for the added precision. We trained this over 20 epochs.

Our criterion is Cross Entropy Loss, which is the standard choice for a multi-class problem like this. It provides a loss function that heavily penalizes the model for being confidently wrong, pushing it to move through the loss surface to find a local minimum—we always hope for a global one, but one can dream. To navigate this surface, we chose the Adam optimizer. While we might have squeezed out more accuracy with something like RMSProp, we unfortunately hit the reality of limited GPU credits.

#### Results and Observations

Our model started off with a validation accuracy of 45 percent and a training loss of 1.4. It rapidly started improving and by the tenth epoch those same metrics stood at 57 percent and 0.96. After, it slowly improved until by the 20th epoch our final validation accuracy was 58.32 percent and our loss was 0.88. An important pattern was that our training loss was decreasing while our validation accuracy was increasing. This makes us think that the model still has a way to go and would likely improve with more epochs. In other words, it is not yet overfitting and reacts to both training and validation data in a positive manner. By positive we mean the validation accuracy is still going up while the training loss is going down.


### The Development Journey: Failures and Iterative Improvements

#### Evolving the CNN: From Basic to "Wide & Deep"
Our journey with the CNN was an iterative process of architectural tuning to handle the sheer volume and complexity of our 33,000-game dataset. We initially started with a standard, shallow 3-layer CNN. While this baseline model learned the rules of the game, it struggled to capture deep strategic patterns and was highly unstable during training.

Realizing the model lacked the "capacity" to process our massive dataset effectively, we overhauled the architecture. First, we expanded the network to 4 convolutional layers, allowing it to build much higher-level abstractions of the board state. However, deeper networks are notoriously hard to train, so we introduced Batch Normalization (BatchNorm2d) after every convolutional layer. This stabilized our gradient flow and allowed the model to learn much faster without the loss exploding.

The true breakthrough, however, was redesigning the filter sizes. Standard image classifiers use 3x3 kernels, but Connect Four is fundamentally about finding four pieces in a line. We modified our third convolutional layer to use a 4x4 kernel (with a padding of 2 to preserve board dimensions). This gave the network a localized "cheat code" to detect winning threats in a single mathematical sweep. 

Finally, to combat the model's tendency to memorize the 33k games as it trained deeper, we added a 30% Dropout rate to our fully connected layers. This forced the network to learn robust, generalizable patterns rather than just memorizing specific board states from the training data.

#### Taming the Transformer: Hyperparameters and the Need for Data
Adapting a Transformer for a grid-based board game proved to be an entirely different challenge. In our early iterations, the Transformer model completely collapsed, getting stuck at a validation accuracy of roughly 16% which mathematically equates to random guessing across 7 columns.

The core issue was "spatial blindness." By flattening the 6x7 board into a sequence of 42 independent tokens, the Transformer lost the inherent geometric grid of the game. To help the model "brute force" its way into understanding these spatial relationships, we experimented with several architectural scaling tweaks. We increased the embedding capacity (d_model) from 128 to 256 dimensions, allowing each board piece to carry a richer mathematical representation. We also deepened the model by increasing the num_encoder_layers from 3 to 4, giving the self-attention mechanism more opportunities to map out the board.

However, the ultimate fix for the Transformer wasn't just code- it was scale. Transformers lack the built-in spatial bias of CNNs and are notoriously data-hungry. The model only began to break out of its 16% rut when we fed it the fully augmented, horizontally-flipped dataset and allowed it to train for extended epochs. By carefully tuning the learning rate (Transformers are highly fragile early in training) and bombarding it with data, the self-attention heads finally learned the spatial structure of Connect Four, pushing the accuracy from 16% up to its final 58.32%.

### Conclusion and Future Work: Intuition vs. Global Strategy

Ultimately, our project demonstrated that the CNN significantly outperformed the Transformer in playing strength. The main difference lies in the nature of the architectures and the volume of data provided.

The CNN utilizes its 4x4 filters to inherently understand spatial relationships, allowing it to perform the optimal move when threats are localized. It does not need massive amounts of data because its architecture already understands which pieces are physically close to each other. On the other hand, the Transformer analyzes the flattened tokens to understand the "global strategy." While this sounds superior in theory, it requires significantly more data to overcome its lack of spatial awareness.

#### Future Improvements
To push our models beyond the intermediate level, several advanced tweaks would be necessary:

**Transformer Optimization:** We could alter the number of self-attention heads or change the final architecture to use a dedicated Class Token ([CLS]) as the input to the MLP, rather than flattening all vectors.

**MCTS-CNN Hybridization (AlphaZero Approach):** To cure the CNN of its "Horizon Effect" and offensive bias, future iterations would not use the CNN as a standalone player. Instead, we would integrate the CNN as a heuristic evaluator inside an MCTS loop. The CNN would provide rapid pattern intuition, while the MCTS provides tactical look-ahead, marrying spatial recognition with deep sequence calculation.
