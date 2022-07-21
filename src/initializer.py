import cnn
import rnn
import torch.optim as optim

def initilize_model(device=None,
                    pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=20,
                    filter_sizes=[8, 16, 32],
                    num_filters=[100, 100, 100],
                    num_classes=21,
                    dropout=0.5,
                    learning_rate=0.01,
                    optimizerName="Adadelta",
                    mome=0.8,
                    modelType="CNN"):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    if modelType=="CNN":
        # Instantiate CNN model
        model = cnn.CNN_NLP(pretrained_embedding=pretrained_embedding,
                            freeze_embedding=freeze_embedding,
                            vocab_size=vocab_size,
                            embed_dim=embed_dim,
                            filter_sizes=filter_sizes,
                            num_filters=num_filters,
                            num_classes=num_classes,
                            dropout=0.5)
    
    elif modelType=="RNN":
        model = rnn.RNNClassifier(vocab_size,
                                  embed_dim,
                                  )
    
    # Send model to `device` (GPU/CPU)
    model.to(device)

    # Instantiate Adadelta optimizer
    if optimizerName == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(),
                                    lr=learning_rate,
                                    rho=0.95)
    elif optimizerName == "SGD":
        optimizer = optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=mome)
    elif optimizerName == "Adam":
        optimizer = optim.Adam(model.parameters(),
                                lr=learning_rate)

    return model, optimizer

