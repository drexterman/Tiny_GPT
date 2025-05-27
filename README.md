# Tiny_GPT
 This is a very basic LLM project trained on tiny shakeshphere

 *Goal* - Create a Language Model based on transformers which autocompletes the given input upto s given token length.

 *Inputs* - A sentence , no. of tokens user wants to be autocompleted

*Outputs* - Autocompleted text

*Training Data* - Little Shakesphere

*Required* - Knowledge of Pytorch, tranformers , optimizers(using AdamW here), Read the 'Attention is all you need' paper by google.

*Assumptions* - Tokens = chars

*Process* 
1) Load the dataset
2) Create a list of all unique characters (65 in this case)
3) Make an encoder & decoder for char <-> int manipulations
4) Train-Test split (90-10 here)
5) Set your batch (B) & block size (T)  (Block size = context length)
6) Make a function to get randomly B(batch) no. of T(block) sized test snippets & return a BxT matrix
7) Make a function to estimate loss of model averaging over 'eval-iters' no. of seq.
8) Define a self-attention head:
   - initialize key,query,value matrix of NxH dimension (N-no. of embedding , H-head size)
   - Also make a upper triangular matrix of TxT dimesion
   - In the forward function you'll get a BxTxN matrix
   - You'll take this & compute the key and query
   - Then there matrix multiplication will give a attention matrix which you'll mask & apply softmax on to get a weight matrix which you'll matrix multiply with value matrix to return a BxTxH matrix.
9) Define a multi-head attention :
   - Initialize the heads
   - Initialize a linear map to convert BxTxH --> BxTxN
   - Take the BxTxH matrix from all N*H heads & concatenate along the H axis
   - Use the projection & return a BxTxN matrix.
10) Define a Feed forward :
    - Initialize a Neural Network with a i)Linear Layer (N -> 4*N) ii)Relu iii) Linear Layer (4*N -> N)
     - Forward the BxTxN dimensional matrix through this.
11) Define a Block :
    -  Initialize the self-attention part, Feedforward & 2 layer norms .
    - In forward pass the layer normed input to Multiheaded attention & add it's output to input.
    - Then pass that to the Feedforward network & again add it's output to the input.
    - Return the operated input.
12)Define the model :
    - Initialize CxN token embedding matrix & TxN position emmbedding matrix
    - Initialize a sequence of L blocks
    - Initialize a Layer Norm & a Linear map from BxTxN --> BxTxC
    - In the forward function take the token & position embeddings of the input & add it to the input .
    - Then push it through the block & the Layer Norm.
    - Then use the Layer map to create a BxTxC matrix containing logits .
    - Then use softmax to return probabilities and loss.
    - It also has a generate function which takes a array, chips to only include BxT elements , forward it through the transformaer 'max_tokens' times to create new characters .
13) Initialize the model
14) Apply an optimizer (AdamW is used here)

# DONE
