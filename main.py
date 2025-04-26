import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import copy
from tqdm import tqdm

# Import from our PyNeuro framework
from pyneuro.core.models.grow_net import GrowNet
from pyneuro.core.losses.loss import CrossEntropyLoss
from pyneuro.datasets.mnist_loader import load_mnist

def plot_confusion_matrix(y_true, y_pred, classes=10):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, range(classes))
    plt.yticks(tick_marks, range(classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def fitness_function(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return correct / total, y_true, y_pred

def main():
    print("🧠 Starting PyNeuro GrowNet with Genetic Optimization 🧠")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 784
    initial_hidden_size = 64
    output_size = 10
    batch_size = 128
    genetic_epochs = 5
    train_epochs = 5
    grow_amount = 8
    
    try:
        # Load MNIST dataset
        print("Loading MNIST dataset...")
        train_loader = load_mnist(train=True, batch_size=batch_size)
        test_loader = load_mnist(train=False, batch_size=batch_size)
        print(f"Dataset loaded with {len(train_loader.dataset)} training samples")
        
        # Initialize population
        population = [GrowNet(input_size, initial_hidden_size, output_size).to(device) 
                     for _ in range(10)]
        loss_fn = CrossEntropyLoss()
        
        # Genetic optimization
        for gen in range(genetic_epochs):
            print(f"\nGeneration {gen+1}/{genetic_epochs}")
            
            # Evaluate population
            fitness_scores = []
            for i, model in enumerate(tqdm(population, desc="Evaluating")):
                fitness, _, _ = fitness_function(model, train_loader, device)
                fitness_scores.append(fitness)
            
            best_idx = np.argmax(fitness_scores)
            best_model = copy.deepcopy(population[best_idx])
            best_fitness = fitness_scores[best_idx]
            print(f"Best fitness: {best_fitness:.4f}")
            
            # Visualize best model
            _, y_true, y_pred = fitness_function(best_model, train_loader, device)
            plot_confusion_matrix(y_true, y_pred)
            
            # Grow network every 2 generations after first
            if gen > 0 and gen % 2 == 1:
                print(f"\n🧠 Growing network by {grow_amount} neurons")
                best_model.grow_brain_cells(grow_amount)
                print(f"New hidden layer size: {best_model.fc1.out_features}")
                
                # Update population with grown model
                population = [copy.deepcopy(best_model) for _ in range(len(population))]
            
            # Create new generation
            new_population = [best_model]
            while len(new_population) < len(population):
                # Tournament selection
                candidates = np.random.choice(len(population), size=3, replace=False)
                parent_idx = candidates[np.argmax([fitness_scores[i] for i in candidates])]
                parent = population[parent_idx]
                
                # Create child through mutation
                child = copy.deepcopy(parent)
                for name, param in child.named_parameters():
                    if 'weight' in name or 'bias' in name:
                        noise = torch.randn_like(param.data) * 0.1
                        param.data += noise
                
                new_population.append(child)
            
            population = new_population
        
        # Final training
        print("\n🚀 Starting final training with best model")
        optimizer = torch.optim.Adam(best_model.parameters(), lr=0.001)
        
        for epoch in range(train_epochs):
            best_model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                data, target = data.to(device), target.to(device)
                data = data.view(-1, input_size)
                
                optimizer.zero_grad()
                output = best_model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            train_acc = correct / total
            val_acc, _, _ = fitness_function(best_model, test_loader, device)
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, "
                  f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        
        # Final evaluation
        print("\n🔥 Final Evaluation")
        test_acc, y_true, y_pred = fitness_function(best_model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.4f}")
        plot_confusion_matrix(y_true, y_pred)
        
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("Using synthetic data instead...")
        
        # Synthetic data fallback
        X_train = torch.randn(1000, input_size).to(device)
        y_train = torch.randint(0, output_size, (1000,)).to(device)
        X_test = torch.randn(200, input_size).to(device)
        y_test = torch.randint(0, output_size, (200,)).to(device)
        
        model = GrowNet(input_size, initial_hidden_size, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = CrossEntropyLoss()
        
        print("\nTraining with synthetic data:")
        for epoch in range(train_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                _, predicted = torch.max(test_output.data, 1)
                accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")
        
        # Plot synthetic results
        model.eval()
        with torch.no_grad():
            _, y_true, y_pred = fitness_function(model, [(X_test, y_test)], device)
        plot_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    main()
