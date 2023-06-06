import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    
    epochs = range(1, 41)
    
    with open('loss.json') as f:
        json_loss = f.read()
        dict_loss = json.loads(json_loss)

    plt.plot(epochs, dict_loss['train_loss'], label='Training Loss')
    plt.plot(epochs, dict_loss['val_loss'], label='Validation Loss')
 
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('plot_epochs.png', bbox_inches='tight')