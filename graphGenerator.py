import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse


def plotFromCSV(csv_file):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file, delimiter=',', header=0)
    
    # Extract the relevant columns
    epochs = data['epoch']
    
    # Plotting Accuracy
    plt.figure()
    plt.plot(epochs, data['train-accuracy'], label='Train Accuracy')
    plt.plot(epochs, data['test-accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    
    # Plotting Loss
    plt.figure()
    plt.plot(epochs, data['train_loss'], label='Train Loss')
    plt.plot(epochs, data['test-loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

# Example usage (ensure you call this in a script or via command line)
plotFromCSV('csv_to_render.csv') #change only this!
