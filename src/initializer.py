import cnn
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
                    mome=0.8):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = cnn.CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=0.5)
    
    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    if optimizerName == "Adadelta":
        optimizer = optim.Adadelta(cnn_model.parameters(),
                                    lr=learning_rate,
                                    rho=0.95)
    elif optimizerName == "SDG":
        optimizer = optim.SDG(cnn_model.parameters(),
                                lr=learning_rate,
                                momentum=mome)
    elif optimizerName == "Adam":
        optimizer = optim.Adam(cnn_model.parameters(),
                                lr=learning_rate)

    return cnn_model, optimizer

